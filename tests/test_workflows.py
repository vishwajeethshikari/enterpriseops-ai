
import httpx
import pytest

from services.workflow_service import create_workflow


@pytest.mark.asyncio
async def test_create_workflow_success(monkeypatch):
    expected = {
        "workflowNumber": "WF-TEST001",
        "status": "PENDING_APPROVAL",
    }

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return expected

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json):
            return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    result = await create_workflow({
        "orderNumber": "ORD-1042",
        "issueDescription": "Investigate delayed delivery",
        "priority": "HIGH",
    })

    assert result["status"] == "PENDING_APPROVAL"
