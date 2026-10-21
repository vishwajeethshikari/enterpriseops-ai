
import logging
from typing import Any

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

WORKFLOWS_API = (
    f"{settings.java_base_url.rstrip('/')}/api/v1/workflows"
)


async def create_workflow(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a workflow proposal in the business backend."""

    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds
        ) as client:
            response = await client.post(
                WORKFLOWS_API,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException as exc:
        logger.warning("Workflow creation timed out")
        raise RuntimeError(
            "The workflow service timed out."
        ) from exc

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Workflow API returned HTTP %s",
            exc.response.status_code,
        )
        raise RuntimeError(
            f"Workflow service returned HTTP {exc.response.status_code}."
        ) from exc

    except httpx.RequestError as exc:
        logger.warning("Unable to reach workflow service")
        raise RuntimeError(
            "Could not connect to the Spring Boot workflow service."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Workflow service returned an invalid response."
        )

    return data


async def get_workflow(
    workflow_number: str,
) -> dict[str, Any]:
    """Retrieve a workflow by its identifier."""

    url = f"{WORKFLOWS_API}/{workflow_number}"

    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise LookupError(
                f"Workflow {workflow_number} was not found."
            ) from exc

        raise RuntimeError(
            f"Workflow service returned HTTP {exc.response.status_code}."
        ) from exc

    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "The workflow service timed out."
        ) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            "Could not connect to the Spring Boot workflow service."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Workflow service returned an invalid response."
        )

    return data
