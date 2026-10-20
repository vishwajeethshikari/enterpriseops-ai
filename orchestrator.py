
import asyncio
import logging
from typing import Any, Awaitable, Callable

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

OrderFetcher = Callable[[str], Awaitable[dict[str, Any]]]
AgentRunner = Callable[[str, dict[str, Any]], dict[str, Any]]
ProposalGenerator = Callable[
    [str, dict[str, Any]],
    Awaitable[Any],
]


async def run_order_workflow(
    order_number: str,
    fetch_order: OrderFetcher,
    run_agents: AgentRunner,
    generate_proposal: ProposalGenerator,
    workflows_api: str,
) -> dict[str, Any]:
    """
    Investigate an order, validate the triage decision,
    and submit a proposed workflow for human approval.

    This function never creates a support ticket directly.
    Spring Boot remains responsible for business operations.
    """

    # 1. Retrieve authoritative order data.
    order = await fetch_order(order_number)

    # 2. Run the existing agents without blocking FastAPI.
    try:
        investigation = await asyncio.to_thread(
            run_agents,
            order_number,
            order,
        )
    except Exception as exc:
        logger.exception(
            "Agent investigation failed for %s",
            order_number,
        )
        raise HTTPException(
            status_code=502,
            detail="The multi-agent investigation failed.",
        ) from exc

    # 3. Validate the overall agent result.
    if not isinstance(investigation, dict):
        raise HTTPException(
            status_code=502,
            detail="The agents returned an invalid result.",
        )

    triage = investigation.get("triage")

    if not isinstance(triage, dict):
        raise HTTPException(
            status_code=502,
            detail="The investigation did not return valid triage data.",
        )

    should_create_ticket = triage.get("should_create_ticket")

    if not isinstance(should_create_ticket, bool):
        raise HTTPException(
            status_code=502,
            detail="The triage decision must be true or false.",
        )

    # 4. Stop safely if the agents do not recommend a ticket.
    if not should_create_ticket:
        return {
            "order_number": order_number,
            "status": "INVESTIGATION_COMPLETED",
            "workflow_created": False,
            "message": (
                "The agents did not recommend a support ticket."
            ),
            "investigation": investigation,
        }

    # 5. Generate a structured ticket proposal.
    proposal = await generate_proposal(
        order_number,
        order,
    )

    if proposal.order_number != order_number:
        raise HTTPException(
            status_code=502,
            detail="The AI proposal contains a mismatched order number.",
        )

    payload = {
        "orderNumber": proposal.order_number,
        "issueDescription": proposal.issue_description,
        "priority": proposal.priority,
    }

    # 6. Submit the proposal to Spring Boot.
    # Spring Boot should persist it as PENDING_APPROVAL.
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                workflows_api,
                json=payload,
            )

    except httpx.TimeoutException as exc:
        logger.exception(
            "Workflow creation timed out for %s",
            order_number,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                "Workflow creation timed out. Check Spring Boot "
                "before retrying to avoid duplicate workflows."
            ),
        ) from exc

    except httpx.RequestError as exc:
        logger.exception(
            "Workflow service unavailable for %s",
            order_number,
        )
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the Spring Boot workflow service.",
        ) from exc

    if response.is_error:
        logger.error(
            "Workflow API returned %s for %s",
            response.status_code,
            order_number,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Spring Boot could not create the workflow.",
                "java_status_code": response.status_code,
            },
        )

    try:
        workflow = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Spring Boot returned invalid workflow JSON.",
        ) from exc

    # 7. Return the actual backend response.
    return {
        "order_number": order_number,
        "status": "WORKFLOW_SUBMITTED",
        "workflow_created": True,
        "message": (
            "Investigation completed and workflow submitted "
            "for human approval."
        ),
        "investigation": investigation,
        "proposal": proposal.model_dump(),
        "workflow": workflow,
    }
