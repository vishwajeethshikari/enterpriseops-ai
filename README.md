# EnterpriseOps AI — Business Workflow Automation Platform

An AI-powered business operations platform that investigates customer orders, generates actionable recommendations, and proposes support tickets using a local Large Language Model (LLM).

## 🚀 Project Overview

EnterpriseOps AI combines a Python AI service with a Spring Boot business backend to assist with customer support and order operations.

The system retrieves real order information, uses an LLM to analyze it, and produces structured investigation results that can help support teams make decisions.

## 🛠️ Tech Stack

- **Backend:** Java, Spring Boot, Spring Data JPA
- **AI Service:** Python, FastAPI
- **LLM:** Ollama, Qwen 2.5 (3B)
- **Database:** PostgreSQL
- **API Communication:** REST APIs
- **Database Migrations:** Flyway

## ✅ Implemented Features

- Retrieve order details through REST APIs.
- Create and retrieve customer support tickets.
- Connect the Python AI service to the Spring Boot backend.
- Investigate orders using a locally running LLM.
- Generate structured JSON containing:
  - Order summary
  - Confirmed facts
  - Missing information
  - Severity assessment
  - Recommended next action
  - Suggested customer message
- Generate AI-proposed support tickets, subject to endpoint testing.

## 🏗️ Architecture

```text
Client / API Request
        |
        v
Python FastAPI AI Service
        |
        +----> Ollama (Qwen 2.5:3B)
        |
        v
Spring Boot Business Service
        |
        v
PostgreSQL Database
```

The Python service handles AI orchestration, while the Spring Boot service manages business data and support-ticket operations.

## ⚙️ Prerequisites

- Java and Maven
- Python 3
- PostgreSQL
- Ollama
- Git

## ▶️ Running the Project

### 1. Start the Spring Boot service

Run the `enterpriseops-business` application from your IDE or using Maven.

The service runs on:

`http://localhost:8080`

### 2. Start Ollama

Ensure Ollama is running and the model is available:

```bash
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

### 3. Start the Python AI service

Navigate to the `enterpriseops-ai` directory, activate your virtual environment, and install the required dependencies.

```bash
python -m pip install fastapi "uvicorn[standard]" httpx ollama
python -m uvicorn main:app --reload --port 8000
```

API documentation:

`http://localhost:8000/docs`

Health endpoint:

`http://localhost:8000/health`

## 🔍 Example API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/orders/{orderNumber}` | Retrieve an order |
| POST | `/api/v1/tickets` | Create a support ticket in the Java service |
| GET | `/api/v1/tickets/{ticketNumber}` | Retrieve a support ticket |
| GET | `/investigate/{order_number}` | Retrieve order data through Python |
| GET | `/ai/investigate/{order_number}` | Generate an AI investigation |
| GET | `/ai/propose-ticket/{order_number}` | Generate an AI ticket proposal |

The Python endpoints run on port `8000`; the Spring Boot endpoints run on port `8080`.

## 🗺️ Planned Features

- Persistent workflow management
- Human approval and rejection of AI-proposed actions
- Reliable ticket creation with duplicate-prevention safeguards
- React dashboard for order investigations and workflow approvals
- Multi-agent orchestration
- Authentication and authorization
- Automated testing, Docker, and deployment

## 🔐 Configuration and Security

- Configure database credentials through environment variables.
- Never commit passwords, API keys, `.env` files, or other secrets.
- Keep local virtual environments and generated build files out of Git.
- Add authentication and authorization before exposing business operations publicly.

## 🎯 Project Goal

Build a portfolio-ready AI engineering application that combines LLM-powered reasoning, business APIs, database persistence, and human-supervised workflow automation.

## 📌 Project Status

**In active development.** The order-management backend, support-ticket APIs, and AI order-investigation foundation have been developed. Workflow automation, the frontend dashboard, and production deployment remain planned until implemented and tested.
