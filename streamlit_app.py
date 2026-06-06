import asyncio
import base64
import os
import time
import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    is_prod = os.getenv("RENDER", "false") == "true" or os.getenv("STREAMLIT_PROD", "false") == "true"
    return inngest.Inngest(app_id="rag_app", is_production=is_prod)


async def send_rag_ingest_event(file_name: str, file_bytes: bytes) -> None:
    client = get_inngest_client()
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_base64": base64_pdf,
                "source_id": file_name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion across cloud services..."):
        file_bytes = uploaded.getvalue()
        asyncio.run(send_rag_ingest_event(uploaded.name, file_bytes))
        time.sleep(0.3)
    st.success(f"Triggered cloud ingestion for: {uploaded.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    if os.getenv("RENDER", "false") == "true" or os.getenv("STREAMLIT_PROD", "false") == "true":
        return "https://api.inngest.com/v1"
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    headers = {}

    if os.getenv("RENDER", "false") == "true" or os.getenv("STREAMLIT_PROD", "false") == "true":
        token = os.getenv("INNGEST_SIGNING_KEY") or os.getenv("INNGEST_EVENT_KEY")
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


# ADDED BACK: The missing asynchronous polling function
async def wait_for_run_output_async(event_id: str, timeout_s: float = 120.0) -> dict:
    start = time.time()
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run execution status evaluated as: {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError("Timed out waiting for context generation output from Inngest.")
        await asyncio.sleep(1.0)


async def run_query_workflow(question_str: str, top_k_val: int) -> dict:
    event_id = await send_rag_query_event(question_str, top_k_val)
    output_data = await wait_for_run_output_async(event_id)
    return output_data


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            try:
                output = asyncio.run(run_query_workflow(question.strip(), int(top_k)))

                answer = output.get("answer", "")
                sources = output.get("sources", [])

                st.subheader("Answer")
                st.write(answer or "(No answer)")
                if sources:
                    st.caption("Sources")
                    for s in sources:
                        st.write(f"- {s}")

            except Exception as e:
                st.error(f"An unexpected query error occurred: {str(e)}")