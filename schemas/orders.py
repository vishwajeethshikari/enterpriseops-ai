
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OrderInvestigationRequest(BaseModel):
    """Validates an order investigation request."""

    model_config = ConfigDict(extra="forbid")

    order_number: str = Field(
        pattern=r"^ORD-\d{4,}$",
        examples=["ORD-1042"],
        description="Unique business order number",
    )


class OrderFacts(BaseModel):
    """Stores verified facts and missing information about an order."""

    model_config = ConfigDict(extra="forbid")

    confirmed_facts: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Facts supported by order data",
    )

    missing_information: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Information not available from the order data",
    )


class OrderInvestigationResponse(BaseModel):
    """Structured result returned by an order investigation."""

    model_config = ConfigDict(extra="forbid")

    order_number: str = Field(
        pattern=r"^ORD-\d{4,}$"
    )

    severity: Literal["LOW", "MEDIUM", "HIGH"]

    summary: str = Field(
        min_length=1,
        max_length=2000,
    )

    facts: OrderFacts
