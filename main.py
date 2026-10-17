
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ollama import AsyncClient
import httpx
import json


app = FastAPI(
    title="EnterpriseOps AI",
    description="AI-powered business order investigation service",
    version="1.0.0"
)

JAVA_API = "http://localhost:8080/api/v1/orders"
JAVA_TICKETS_API = "http://localhost:8080/api/v1/tickets"
OLLAMA_MODEL = "qwen2.5:3b"


# -----------------------------------------
# 1. Structured AI response models
# -----------------------------------------

class OrderFacts(BaseModel):
    confirmed_facts: list[str]
    missing_information: list[str]


class AIOrderInvestigation(BaseModel):
    order_number: str
    severity: str
    order_summary: str
    facts: OrderFacts
    recommended_action: str
    customer_message: str


# -----------------------------------------
# 2. Health check
# -----------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "enterpriseops-ai"
    }


# -----------------------------------------
# 3. Retrieve an order from Spring Boot
# -----------------------------------------

async def fetch_order(order_number: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{JAVA_API}/{order_number}"
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Java service request timed out"
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Java service"
        ) from exc

    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail="Order not found"
        )

    if response.is_error:
        raise HTTPException(
            status_code=502,
            detail="Java service returned an error"
        )

    return response.json()


# -----------------------------------------
# 4. Basic order investigation
# -----------------------------------------

@app.get("/investigate/{order_number}")
async def investigate_order(order_number: str):
    order = await fetch_order(order_number)

    return {
        "order": order,
        "investigation_status": "ORDER_DATA_RETRIEVED",
        "message": "Order retrieved successfully."
    }


# -----------------------------------------
# 5. AI-powered structured investigation
# -----------------------------------------

@app.get(
    "/ai/investigate/{order_number}",
    response_model=AIOrderInvestigation
)
async def ai_investigate_order(order_number: str):
    # Step 1: Retrieve real order data
    order = await fetch_order(order_number)

    # Step 2: Prepare instructions for the AI
    prompt = f"""
You are an order investigation assistant for EnterpriseOps.

Analyze only the supplied order data.

Rules:
- Use only facts supported by the supplied data.
- Do not invent the cause of a delay.
- Do not assume the order creation date is the purchase date.
- Separate confirmed facts from missing information.
- If the cause of a delay is unknown, say so.
- Recommend a practical next investigation step.
- Draft a polite customer update without inventing delivery dates.
- Severity must be one of: LOW, MEDIUM, HIGH.
- Return all required fields in the requested JSON format.

Order data:
{json.dumps(order, indent=2)}

Required JSON structure:
{{
  "order_number": "string",
  "severity": "LOW, MEDIUM, or HIGH",
  "order_summary": "string",
  "facts": {{
    "confirmed_facts": ["string"],
    "missing_information": ["string"]
  }},
  "recommended_action": "string",
  "customer_message": "string"
}}
"""

    # Step 3: Ask Ollama for structured JSON
    try:
        client = AsyncClient()

        result = await client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format=AIOrderInvestigation.model_json_schema(),
            options={
                "temperature": 0
            }
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not generate AI analysis. "
                "Check that Ollama is running and the model is available."
            )
        ) from exc

    # Step 4: Validate the generated JSON
    try:
        investigation = AIOrderInvestigation.model_validate_json(
            result.message.content
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The AI returned an invalid investigation response"
        ) from exc

    # Step 5: Return validated structured data
    return investigation

from typing import Literal

from pydantic import BaseModel, Field


class CreateTicketRequest(BaseModel):
    orderNumber: str = Field(
        min_length=1,
        description="Existing order number, such as ORD-1042"
    )
    issueDescription: str = Field(
        min_length=5,
        description="Description of the customer issue"
    )
    priority: Literal["LOW", "MEDIUM", "HIGH"]


@app.post("/tickets", status_code=201)
async def create_support_ticket(request: CreateTicketRequest):
    # Verify that the referenced order exists.
    order = await fetch_order(request.orderNumber)

    payload = {
        "orderNumber": request.orderNumber,
        "issueDescription": request.issueDescription,
        "priority": request.priority
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                JAVA_TICKETS_API,
                json=payload
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "Ticket creation timed out. Check the Java service "
                "before retrying to avoid creating a duplicate."
            )
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the Java ticket service"
        ) from exc

    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "message": "Java service could not create the ticket",
                "response": response.text
            }
        )

    return {
        "message": "Support ticket created successfully",
        "order": order,
        "ticket": response.json()
    }

