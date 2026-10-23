
import pytest
import agents


ORDER = {
    "orderNumber": "ORD-1042",
    "status": "DELAYED",
    "expectedDelivery": "2026-10-25",
}


def test_multi_agent_returns_proposal_without_creating_ticket(
    monkeypatch,
):
    def fake_investigator(order):
        return {
            "order_number": "ORD-1042",
            "order_summary": "Order is delayed",
            "confirmed_facts": ["Status is DELAYED"],
            "missing_information": ["Reason for delay"],
            "recommended_action": "Verify shipment status",
        }

    def fake_policy(order):
        return {
            "answer": "Verify shipment status before making promises.",
            "policy_sufficient": True,
            "sources": [],
        }

    def fake_triage(order, investigation, policy_result):
        assert investigation["order_number"] == "ORD-1042"
        assert policy_result["policy_sufficient"] is True

        return {
            "should_create_ticket": True,
            "priority": "MEDIUM",
            "issue_description": "Investigate delayed delivery",
            "reason": "The order is delayed",
            "requires_human_approval": True,
        }

    monkeypatch.setattr(
        agents, "order_investigator_agent", fake_investigator
    )
    monkeypatch.setattr(agents, "policy_agent", fake_policy)
    monkeypatch.setattr(agents, "triage_agent", fake_triage)

    result = agents.run_multi_agent_investigation(
        "ORD-1042", ORDER
    )

    assert result["order_number"] == "ORD-1042"
    assert result["workflow_status"] == "PROPOSAL_ONLY"
    assert "order_investigator" in result["agents"]
    assert "policy_agent" in result["agents"]
    assert "support_triage" in result["agents"]
    assert result["agents"]["support_triage"]["requires_human_approval"] is True
    assert "No support ticket was created" in result["next_step"]


def test_multi_agent_rejects_mismatched_order_number():
    with pytest.raises(
        RuntimeError,
        match="The returned order does not match the requested order",
    ):
        agents.run_multi_agent_investigation(
            "ORD-9999", ORDER
        )


def test_triage_rejects_invalid_priority(monkeypatch):
    monkeypatch.setattr(
        agents,
        "ask_llm_json",
        lambda prompt: {
            "should_create_ticket": True,
            "priority": "URGENT",
            "issue_description": "Investigate delayed delivery",
            "reason": "Order is delayed",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Triage Agent returned an invalid priority",
    ):
        agents.triage_agent(
            ORDER,
            {"order_summary": "Order is delayed"},
            {"answer": "Verify status", "sources": []},
        )


def test_triage_requires_valid_issue_description(monkeypatch):
    monkeypatch.setattr(
        agents,
        "ask_llm_json",
        lambda prompt: {
            "should_create_ticket": True,
            "priority": "MEDIUM",
            "issue_description": "Oops",
            "reason": "Order is delayed",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Triage Agent returned an invalid issue description",
    ):
        agents.triage_agent(
            ORDER,
            {"order_summary": "Order is delayed"},
            {"answer": "Verify status", "sources": []},
        )
