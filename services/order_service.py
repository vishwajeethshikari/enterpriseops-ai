
import logging
from typing import Any

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

ORDERS_API = f"{settings.java_base_url.rstrip('/')}/api/v1/orders"


async def get_order(order_number: str) -> dict[str, Any]:
    """Fetch an order from the Spring Boot business API."""

    url = f"{ORDERS_API}/{order_number}"

    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException as exc:
        logger.warning("Order API timed out for %s", order_number)
        raise RuntimeError("The order service timed out.") from exc

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise LookupError(
                f"Order {order_number} was not found."
            ) from exc

        logger.exception("Order API returned an HTTP error")
        raise RuntimeError(
            f"Order service returned HTTP {exc.response.status_code}."
        ) from exc

    except httpx.RequestError as exc:
        logger.warning("Unable to reach the order service")
        raise RuntimeError(
            "Could not connect to the Spring Boot order service."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("Order service returned an invalid response.")

    returned_order_number = data.get("orderNumber")

    if returned_order_number != order_number:
        raise RuntimeError(
            "The order service returned a different order number."
        )

    return data
