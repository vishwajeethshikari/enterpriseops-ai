
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkflowProposalRequest(BaseModel):
    """Validates a proposed business workflow."""

    model_config = ConfigDict(extra="forbid")

    order_number: str = Field(
        pattern=r"^ORD-\d{4,}$",
        examples=["ORD-1042"],
    )

    issue_description: str = Field(
        min_length=5,
        max_length=2000,
        description="Explanation of the issue requiring investigation",
    )

    priority: Literal["LOW", "MEDIUM", "HIGH"]


class WorkflowStatusResponse(BaseModel):
    """Represents the status of a business workflow."""

    model_config = ConfigDict(extra="allow")

    workflow_number: str | None = None

    status: str = Field(
        min_length=1,
        description="Current workflow status",
    )

    approval_decision: str | None = None

    ticket: dict | None = None
