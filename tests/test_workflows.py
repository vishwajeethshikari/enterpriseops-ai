
import httpx
import pytest

from services.workflow_service import create_workflow


PAYLOAD = {
    "orderNumber": "ORD-1042",
    "issueDescription": "Investigate delayed delivery",
    "priority": "HIGH",
}


class MockResponse:
    def __init__(self, data=None, status_code=201):
        self.data = data if data is not None else {
            "workflowNumber": "WF-TEST001",
            "status": "PENDING_APPROVAL",
        }
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://test/workflows")
            response = httpx.Response(
                self.status_code,
                request=request,
            )
            raise httpx.HTTPStatusError(
                "HTTP error",
                request=request,
                response=response,
            )

    def json(self):
        return self.data


class MockClient:
    def __init__(self, *args, response=None, error=None, **kwargs):
        self.response = response
        self.error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, json):
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_create_workflow_success(monkeypatch):
    expected = {
        "workflowNumber": "WF-TEST001",
        "status": "PENDING_APPROVAL",
    }

    def client_factory(*args, **kwargs):
        return MockClient(
            response=MockResponse(expected)
        )

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    result = await create_workflow(PAYLOAD)

    assert result == expected


@pytest.mark.asyncio
async def test_create_workflow_handles_http_error(monkeypatch):
    def client_factory(*args, **kwargs):
        return MockClient(
            response=MockResponse(status_code=500)
        )

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    with pytest.raises(
        RuntimeError,
        match="Workflow service returned HTTP 500",
    ):
        await create_workflow(PAYLOAD)


@pytest.mark.asyncio
async def test_create_workflow_handles_timeout(monkeypatch):
    def client_factory(*args, **kwargs):
        return MockClient(
            error=httpx.TimeoutException("Request timed out")
        )

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    with pytest.raises(
        RuntimeError,
        match="The workflow service timed out",
    ):
        await create_workflow(PAYLOAD)


@pytest.mark.asyncio
async def test_create_workflow_handles_connection_error(monkeypatch):
    request = httpx.Request(
        "POST",
        "http://test/workflows",
    )

    def client_factory(*args, **kwargs):
        return MockClient(
            error=httpx.ConnectError(
                "Connection failed",
                request=request,
            )
        )

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    with pytest.raises(
        RuntimeError,
        match="Could not connect to the Spring Boot workflow service",
    ):
        await create_workflow(PAYLOAD)


@pytest.mark.asyncio
async def test_create_workflow_rejects_non_dictionary_response(
    monkeypatch,
):
    def client_factory(*args, **kwargs):
        return MockClient(
            response=MockResponse(["unexpected", "list"])
        )

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    with pytest.raises(
        RuntimeError,
        match="Workflow service returned an invalid response",
    ):
        await create_workflow(PAYLOAD)

