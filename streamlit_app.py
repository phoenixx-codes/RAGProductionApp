import asyncio
import base64
import os
import time
import inngest
import requests
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    is_prod = os.getenv("RENDER", "false") == "true" or os.getenv("STREAMLIT_PROD", "false") == "true"
    return inngest.Inngest(app_id="rag_app", is_production=is_prod)


async def send_rag_ingest_event(file_name: str, secure_signed_url: str) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_url": secure_signed_url,
                "source_id": file_name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    safe_filename = uploaded.name.replace(" ", "_")
    # Tracking the active file context across browser refreshes/reruns
    st.session_state["active_file"] = safe_filename

    if st.session_state.get("last_uploaded") != safe_filename:
        with st.spinner("Uploading file securely to private cloud storage..."):
            supabase = get_supabase_client()
            file_bytes = uploaded.getvalue()

            try:
                supabase.storage.from_("pdfs").upload(
                    path=safe_filename,
                    file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )

                sign_response = supabase.storage.from_("pdfs").create_signed_url(
                    path=safe_filename,
                    expires_in=900
                )
                secure_url = sign_response["signedURL"]

                asyncio.run(send_rag_ingest_event(safe_filename, secure_url))
                st.session_state["last_uploaded"] = safe_filename
                time.sleep(0.3)
                st.success(f"Successfully processed and triggered secure cloud ingestion!")

            except Exception as upload_err:
                st.error(f"Cloud storage pipeline failure: {str(upload_err)}")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int, source_id: str) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
                "source_id": source_id,
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
    except Exception as e:
        print(f"Polling HTTP network exception encountered: {str(e)}")
        return []


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


async def run_query_workflow(question_str: str, top_k_val: int, source_id: str) -> dict:
    event_id = await send_rag_query_event(question_str, top_k_val, source_id)
    output_data = await wait_for_run_output_async(event_id)
    return output_data


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        active_file = st.session_state.get("active_file", "")
        if not active_file:
            st.warning("Please upload a PDF file first to establish a search context context.")
        else:

            status_holder = st.empty()

            with status_holder.container():
                st.info(" Query dispatched to cloud infrastructure. Processing semantics...")

            try:
                output = asyncio.run(run_query_workflow(question.strip(), int(top_k), active_file))

                answer = output.get("answer", "")
                sources = output.get("sources", [])

                # Clear the info alert box out completely before printing the results
                status_holder.empty()

                st.subheader("Answer")
                st.write(answer or "(No answer)")
                if sources:
                    st.caption("Sources")
                    for s in sources:
                        st.write(f"- {s}")

            except Exception as e:
                status_holder.empty()
                # Catching timeouts or network spikes gracefully
                if "TimeoutError" in str(type(e)) or "timeout" in str(e).lower():
                    st.warning(
                        " The server is taking a moment to compile a response for this dense context. Please click 'Ask' again in 5 seconds—the backend is already warm!")
                else:
                    st.error(f"An unexpected query error occurred: {str(e)}")