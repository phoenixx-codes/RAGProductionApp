import datetime
import logging
import os
import tempfile
import uuid
import requests
from fastapi import FastAPI
from groq import Groq
from inngest import Inngest, Context, TriggerEvent
import inngest.fast_api
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=os.getenv("RENDER", "false") == "true",
    serializer=inngest.PydanticSerializer()
)


def _load_step(ctx: Context) -> RAGChunkAndSrc:
    pdf_url = ctx.event.data.get("pdf_url")
    source_id = ctx.event.data.get("source_id", "document.pdf")

    if not pdf_url:
        raise ValueError("Missing 'pdf_url' string parameter in event data.")

    response = requests.get(pdf_url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch document from cloud storage. Status: {response.status_code}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_pdf.write(chunk)
        temp_path = temp_pdf.name

    try:
        chunks = load_and_chunk_pdf(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return RAGChunkAndSrc(chunks=chunks, source_id=source_id)


def _upsert_step(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
    chunks = chunks_and_src.chunks
    source_id = chunks_and_src.source_id
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
    QdrantStorage().upsert(ids, vecs, payloads)
    return RAGUpsertResult(ingested=len(chunks))


def _search_step(question: str, top_k: int = 5, source_id: str = None) -> RAGSearchResult:
    query_vec = embed_texts([question])[0]
    store = QdrantStorage()
    # Route tracking constraint variables straight into your database engine instance
    found = store.search(query_vec, top_k, source_id=source_id)
    return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])


@inngest_client.create_function(
    fn_id="rag_ingest_pdf",
    trigger=TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: Context):
    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load_step(ctx),
        output_type=RAGChunkAndSrc
    )
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert_step(chunks_and_src),
        output_type=RAGUpsertResult
    )
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="rag_query_pdf_ai",
    trigger=TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    source_id = ctx.event.data.get("source_id", None)  # Parse the target metadata scope reference

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search_step(question, top_k, source_id=source_id),
        output_type=RAGSearchResult
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    def _call_llm():
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content

    answer = await ctx.step.run("llm-answer", _call_llm)

    return {
        "answer": answer.strip(),
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }


app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])