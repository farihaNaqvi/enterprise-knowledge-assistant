# Enterprise Knowledge Assistant 🤖

A professional RAG (Retrieval-Augmented Generation) system built with **FastAPI** and **LangChain**. This assistant allows users to upload PDF documents and query them using OpenAI's LLMs with high accuracy and source citations.

## 🚀 Features
- **PDF Ingestion:** Automated text extraction and chunking using `PyPDF` and `RecursiveCharacterTextSplitter`.
- **Vector Search:** High-performance similarity search powered by `ChromaDB`.
- **RAG Chain:** Advanced retrieval logic using `langchain-classic` for stable enterprise performance.
- **REST API:** Fully documented interactive API endpoints via FastAPI (Swagger UI).

## 🛠️ Tech Stack
- **Framework:** FastAPI
- **AI Orchestration:** LangChain (v1.x)
- **Database:** ChromaDB (Vector Store)
- **Environment Management:** Poetry

## 📦 Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/farihaNaqvi/enterprise-knowledge-assistant.git
2. **Install dependencies:**

   ```Bash
   poetry install
3. **Configure Environment:** Create a .env file and add your OPENAI_API_KEY.

4. **Run the Application:**

   ```Bash
   poetry run uvicorn app.main:app --reload
