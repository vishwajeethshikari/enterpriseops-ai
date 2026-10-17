
from typing import Literal
import json

import httpx
from fastapi import FastAPI, HTTPException
from ollama import AsyncClient
from pydantic import BaseModel, Field


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

app = FastAPI(
    title="EnterpriseOps AI",
    description="AI-powered business order investigation and workflow automation",
    version="1.1.0",
)

JAVA_API = "http://localhost:8080/api/v1/orders"
JAVA_TICKETS_API = "http://localhost:8080/api/v1/tickets"
JAVA_WORKFLOWS_API = "http://localhost:8080/api/v1/workflows"

OLLAMA_MODEL = "qwen2.5:3b"


# =========================================================
# 2. PYDANTIC RESPONSE AND REQUEST MODELS
# =========================================================

class OrderFacts(BaseModel):
    confirmed_facts: list[str]
    missing_information: list[str]


class AIOrderInvestigation(BaseModel):
    order_number: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    order_summary: str
    facts: OrderFacts
    recommended_action: str
    customer_message: str


class CreateTicketRequest(BaseModel):
    orderNumber: str = Field(min_length=1)
    issueDescription: str = Field(min_length=5, max_length=2000)
    priority: Literal["LOW", "MEDIUM", "HIGH"]


class TicketProposal(BaseModel):
    order_number: str
    issue_description: str = Field(min_length=5, max_length=2000)
    priority: Literal["LOW", "MEDIUM", "HIGH"]
    reason: str


# =========================================================
# 3. HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "enterpriseops-ai",
    }


# =========================================================
# 4. RETRIEVE ORDER FROM SPRING BOOT
# =========================================================

async def fetch_order(order_number: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{JAVA_API}/{order_number}"
            )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Java order service request timed out",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the Java order service",
        ) from exc

    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail="Order not found",
        )

    if response.is_error:
        raise HTTPException(
            status_code=502,
            detail="Java order service returned an error",
        )

    return response.json()


# =========================================================
# 5. BASIC ORDER INVESTIGATION
# =========================================================

@app.get("/investigate/{order_number}")
async def investigate_order(order_number: str):
    order = await fetch_order(order_number)

    return {
        "order": order,
        "investigation_status": "ORDER_DATA_RETRIEVED",
        "message": "Order retrieved successfully.",
    }


# =========================================================
# 6. OLLAMA HELPER FOR STRUCTURED AI RESPONSES
# =========================================================

async def generate_structured_response(
    prompt: str,
    response_model: type[BaseModel],
) -> BaseModel:
    try:
        result = await AsyncClient().chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=response_model.model_json_schema(),
            options={"temperature": 0},
        )

        return response_model.model_validate_json(
            result.message.content
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not generate a valid AI response. "
                "Check Ollama and the configured model."
            ),
        ) from exc


# =========================================================
# 7. AI-POWERED ORDER INVESTIGATION
# =========================================================

@app.get(
    "/ai/investigate/{order_number}",
    response_model=AIOrderInvestigation,
)
async def ai_investigate_order(order_number: str):
    order = await fetch_order(order_number)

    prompt = f"""
You are an order investigation assistant for EnterpriseOps.

Analyze only the supplied order data.

Rules:
- Use only facts supported by the supplied data.
- Do not invent the cause of a delay.
- Do not assume the order creation date is the purchase date.
- Separate confirmed facts from missing information.
- If the delay cause is unknown, say so.
- Recommend a practical next investigation step.
- Draft a polite customer update without inventing delivery dates.
- Severity must be LOW, MEDIUM, or HIGH.
- Use the exact order number supplied below.

Requested order number: {order_number}

Order data:
{json.dumps(order, indent=2)}
"""

    investigation = await generate_structured_response(
        prompt,
        AIOrderInvestigation,
    )

    if investigation.order_number != order_number:
        raise HTTPException(
            status_code=502,
            detail="AI returned a mismatched order number",
        )

    return investigation


# =========================================================
# 8. GENERATE AN AI SUPPORT-TICKET PROPOSAL
# =========================================================

async def generate_ticket_proposal(
    order_number: str,
    order: dict,
) -> TicketProposal:
    prompt = f"""
You are an EnterpriseOps customer-support assistant.

Propose a support ticket based only on the supplied order data.

Rules:
- Use the exact requested order number.
- Describe the issue clearly and actionably.
- Do not invent the cause of a delay.
- Do not invent customer details or delivery dates.
- Do not claim that a ticket or workflow has been created.
- Choose LOW, MEDIUM, or HIGH priority.
- Explain the reason for the selected priority.
- The issue description must be between 5 and 2000 characters.

Requested order number: {order_number}

Order data:
{json.dumps(order, indent=2)}
"""

    proposal = await generate_structured_response(
        prompt,
        TicketProposal,
    )

    if proposal.order_number != order_number:
        raise HTTPException(
            status_code=502,
            detail="AI returned a mismatched order number",
        )

    return proposal


@app.get(
    "/ai/propose-ticket/{order_number}",
    response_model=TicketProposal,
)
async def propose_ticket(order_number: str):
    order = await fetch_order(order_number)

    return await generate_ticket_proposal(
        order_number,
        order,
    )


# =========================================================
# 9. CREATE A SUPPORT TICKET DIRECTLY
# =========================================================

@app.post("/tickets", status_code=201)
async def create_support_ticket(request: CreateTicketRequest):
    order = await fetch_order(request.orderNumber)

    payload = {
        "orderNumber": request.orderNumber,
        "issueDescription": request.issueDescription,
        "priority": request.priority,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                JAVA_TICKETS_API,
                json=payload,
            )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "Ticket creation timed out. Check the Java service "
                "before retrying to avoid creating a duplicate."
            ),
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the Java ticket service",
        ) from exc

    if response.is_error:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Java service could not create the ticket",
                "response": response.text,
            },
        )

    return {
        "message": "Support ticket created successfully",
        "order": order,
        "ticket": response.json(),
    }


# =========================================================
# 10. SEND AN AI PROPOSAL TO THE WORKFLOW ENGINE
# =========================================================

@app.post("/ai/create-workflow/{order_number}")
async def create_workflow_from_ai(order_number: str):
    # Step 1: Retrieve authoritative order data.
    order = await fetch_order(order_number)

    # Step 2: Generate and validate the AI proposal.
    proposal = await generate_ticket_proposal(
        order_number,
        order,
    )

    # Step 3: Match CreateWorkflowRequest.java.
    payload = {
        "orderNumber": proposal.order_number,
        "issueDescription": proposal.issue_description,
        "priority": proposal.priority,
    }

    # Step 4: Ask Spring Boot to persist the workflow.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                JAVA_WORKFLOWS_API,
                json=payload,
            )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "Workflow creation timed out. Verify the workflow "
                "in Spring Boot before retrying."
            ),
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the Java workflow service",
        ) from exc

    if response.is_error:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Java could not create the workflow",
                "java_status_code": response.status_code,
                "response": response.text,
            },
        )

    # Step 5: Return the actual workflow returned by Java.
    return {
        "message": "AI proposal submitted to the workflow service",
        "order_number": order_number,
        "proposal": proposal.model_dump(),
        "workflow": response.json(),
    }
