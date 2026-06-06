# 📄 Asynchronous Production RAG Application

A production-grade, serverless Retrieval-Augmented Generation (RAG) pipeline built to ingest, chunk, embed, and query large PDF documents asynchronously. 

This project decouples the heavy vector computational architecture from the user interface using event-driven microservices to maintain a lightning-fast UI and ensure absolute memory stability.

## 🚀 Live Links

* **Live Interactive UI:** [https://rag-pdf-assistant-1.streamlit.app/](https://rag-pdf-assistant-1.streamlit.app/)
* **Production API Gateway (Render):** [https://ragproductionapp-1-hnt9.onrender.com](https://ragproductionapp-1-hnt9.onrender.com)
* **Interactive API Documentation:** [https://ragproductionapp-1-hnt9.onrender.com/docs](https://ragproductionapp-1-hnt9.onrender.com/docs)

---

## 🏗️ System Architecture

Rather than executing extraction and LLM generations directly inside the frontend thread, this application implements a highly scalable, event-driven pattern:

1. **Frontend (Streamlit Community Cloud):** Accepts PDF uploads, converts files to safe Base64 payloads, publishes transactional payloads to the cloud event bus, and cleanly polls for execution status.
2. **Orchestration Layer (Inngest Cloud):** Acts as a durable execution event broker, queueing requests and triggering distributed steps to manage retries and pipeline state safely.
3. **Compute Core (FastAPI on Render):** Processes computationally intensive tasks out-of-band: handles ephemeral PDF decoding, reads tables/text via LlamaIndex, upserts vectors to the cloud index, and generates structured contexts.
4. **Vector Database (Qdrant Cloud):** Manages high-performance distance metric indices, isolated schemas, and payload properties across 384-dimensional spaces.
5. **LLM Engine (Groq Cloud API):** Utilizes `llama-3.3-70b-versatile` to process contextual data structures and emit highly deterministic answers within milliseconds.

---

## 🛠️ Technology Toolkit

* **Core Frameworks:** FastAPI, Streamlit, Inngest Python SDK
* **RAG Parsing & Storage:** LlamaIndex (Core & File Readers), Qdrant Client
* **Environment & Matrix Processing:** Uvicorn, Requests, Pydantic, Python-Dotenv
* **Dependency & Build Automation:** `uv` (Astral's ultra-fast package installer), Docker, Multi-stage Debian-slim builds, Supervisor (Process Manager)

---

## 📦 Local Installation & Setup

This repository uses Astral's `uv` for seamless, lightning-fast virtual environment management.

### 1. Clone the Repository
```bash
git clone [https://github.com/phoenixx-codes/RAGProductionApp.git](https://github.com/phoenixx-codes/RAGProductionApp.git)
cd RAGProductionApp