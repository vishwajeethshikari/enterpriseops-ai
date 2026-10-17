# EnterpriseOps AI — AI-Powered Business Workflow Automation

EnterpriseOps AI is an AI-powered business operations platform that investigates customer orders, retrieves relevant support policies, generates actionable recommendations, and proposes support-ticket workflows using locally hosted Large Language Models (LLMs).

The project combines a Python AI service with a Spring Boot business backend to build a foundation for reliable, human-supervised AI automation.

## 🚀 Project Overview

EnterpriseOps AI helps customer-support teams investigate order issues and make informed decisions.

The system retrieves real order information from the Java backend, uses an LLM to analyze it, and applies Retrieval-Augmented Generation (RAG) to answer questions using relevant support-policy documents.

The long-term goal is to build a multi-agent business workflow automation platform with human approval and auditable actions.

## 🛠️ Tech Stack

- **Backend:** Java, Spring Boot, Spring Data JPA
- **AI Service:** Python, FastAPI
- **LLMs:** Ollama, Qwen 2.5 (3B)
- **Embeddings:** Ollama, `nomic-embed-text`
- **RAG / Vector Database:** ChromaDB
- **Business Database:** PostgreSQL
- **Database Migrations:** Flyway
- **API Communication:** REST APIs
- **Version Control:** Git and GitHub

## ✅ Implemented Features

### Business Operations
- Retrieve order details through REST APIs.
- Create and retrieve customer support tickets.
- Connect the Python AI service to the Spring Boot backend.
- Investigate orders using a locally hosted LLM.
- Generate structured investigation results containing:
  - Order summary
  - Confirmed facts
  - Missing information
  - Severity assessment
  - Recommended next action
  - Suggested customer message

### Retrieval-Augmented Generation (RAG)
- Load support-policy documents written in Markdown.
- Split documents into smaller text chunks.
- Generate text embeddings using Ollama's `nomic-embed-text` model.
- Store embeddings and policy chunks persistently in ChromaDB.
- Retrieve relevant policy passages using semantic similarity search.
- Generate answers using Qwen with retrieved policy passages as context.
- Return the retrieved source passages alongside the generated answer.

### AI-Assisted Support
- Generate AI-proposed support-ticket details.
- Prepare proposed workflows for integration with human approval.

> Features are being developed and tested incrementally. See the planned features below for functionality that is not yet fully integrated or verified.

## 🏗️ Architecture

```text
                    Client / API Request
                            |
              +-------------+-------------+
              |                           |
              v                           v
      Python FastAPI                Spring Boot
       AI Service                 Business Service
              |                           |
       +------+-------+                   v
       |              |               PostgreSQL
       v              v
    Ollama         ChromaDB
  Qwen 2.5:3B    Policy Vectors
       |
       v
  AI-Generated Answers
  with Retrieved Evidence
```

The Spring Boot service manages business data and operations. The Python service handles AI orchestration, embeddings, policy retrieval, and LLM-powered responses.

## 📁 Project Structure

```text
enterpriseops-ai/
├── Knowledge/
│   ├── rag.py
│   └── support_policy.md
├── chroma_db/              # Generated locally; not committed
├── main.py                 # FastAPI AI service
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Prerequisites

- Java and Maven
- Python 3
- PostgreSQL
- Git
- Ollama

## ▶️ Running the Project

### 1. Start the Spring Boot Business Service

Run the `enterpriseops-business` application from your IDE or using Maven.

The service is expected to run at:

```text
http://localhost:8080
```

### 2. Start Ollama and Download the Models

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Ensure Ollama is running before using the AI features.

### 3. Set Up the Python Environment

From the `enterpriseops-ai` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 4. Index Support Policies

Run the RAG ingestion command:

```powershell
python Knowledge/rag.py ingest
```

This reads Markdown policy files, generates embeddings, and stores them in ChromaDB.

### 5. Search Support Policies

```powershell
python Knowledge/rag.py search "How should we handle delayed orders?"
```

### 6. Generate an Answer Using RAG

```powershell
python Knowledge/rag.py ask "Can we promise a delivery date for a delayed order?"
```

The command retrieves relevant policy passages and asks Qwen to answer using the retrieved context.

### 7. Start the FastAPI AI Service

```powershell
python -m uvicorn main:app --reload --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

Health endpoint:

```text
http://localhost:8000/health
```

## 🔍 Example API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check AI service health |
| GET | `/investigate/{order_number}` | Retrieve order data through Python |
| GET | `/ai/investigate/{order_number}` | Generate an AI order investigation |
| GET | `/ai/propose-ticket/{order_number}` | Generate an AI ticket proposal |
| POST | `/tickets` | Request support-ticket creation through Java |

The Java business service also provides order and support-ticket APIs on port `8080`.

The command-line RAG functionality is implemented in `Knowledge/rag.py`. A dedicated FastAPI RAG endpoint remains to be integrated and tested.

## 🗺️ Planned Features

- Integrate RAG questions and answers into FastAPI.
- Add LLM tool calling for controlled business operations.
- Build specialized agents for order investigation, policy analysis, and support triage.
- Implement coordinator-based multi-agent orchestration.
- Integrate workflow approval and rejection into the AI workflow.
- Add duplicate-prevention safeguards and idempotency.
- Build a React dashboard for investigations and approvals.
- Add authentication, authorization, audit logs, and automated tests.
- Containerize and deploy the application.

## 🔐 Configuration and Security

- Store database credentials in environment variables.
- Never commit passwords, API keys, `.env` files, or other secrets.
- Keep virtual environments and generated vector database files out of Git.
- Treat LLM-generated outputs as recommendations until validated.
- Require appropriate authorization and human approval for sensitive business actions.

## 🎯 Project Goal

Build a portfolio-ready AI engineering application that combines LLM-powered reasoning, RAG, business APIs, persistent workflows, and human-supervised automation.

## 📌 Project Status

**Status: Active development**

The project currently includes a Python AI service, Spring Boot business APIs, local LLM-based order investigation, and a tested command-line RAG pipeline using ChromaDB and Ollama. FastAPI integration of RAG, tool calling, multi-agent orchestration, and the frontend dashboard remain in progress.

