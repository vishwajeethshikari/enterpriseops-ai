
import pytest
from pydantic import ValidationError

from schemas.orders import (
    OrderFacts,
    OrderInvestigationRequest,
)
from schemas.workflows import WorkflowProposalRequest


def test_valid_order_number():
    request = OrderInvestigationRequest(
        order_number="ORD-1042"
    )
    assert request.order_number == "ORD-1042"


def test_invalid_order_number():
    with pytest.raises(ValidationError):
        OrderInvestigationRequest(order_number="INVALID")


def test_order_facts_require_confirmed_facts():
    with pytest.raises(ValidationError):
        OrderFacts(confirmed_facts=[])


def test_valid_workflow_proposal():
    proposal = WorkflowProposalRequest(
        order_number="ORD-1042",
        issue_description="Investigate delayed delivery",
        priority="HIGH",
    )
    assert proposal.priority == "HIGH"


def test_invalid_workflow_priority():
    with pytest.raises(ValidationError):
        WorkflowProposalRequest(
            order_number="ORD-1042",
            issue_description="Investigate delayed delivery",
            priority="URGENT",
        )
