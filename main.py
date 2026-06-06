
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
#from inngest.experimental.ai.groq import Adapter
from dotenv import load_dotenv
import uuid
import os
import datetime
from groq import Groq
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
import base64
import tempfile


load_dotenv()


inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=os.getenv("RENDER", "false") == "true",
    serializer=inngest.PydanticSerializer()
)



def _load_step(ctx: inngest.Context) -> RAGChunkAndSrc:
    pdf_path = ctx.event.data.get("pdf_path")
    if not pdf_path:
        raise ValueError("Missing 'pdf_path' in event data.")
    source_id = ctx.event.data.get("source_id", pdf_path)
    chunks = load_and_chunk_pdf(pdf_path)
    return RAGChunkAndSrc(chunks=chunks, source_id=source_id)def _load_step(ctx: inngest.Context) -> RAGChunkAndSrc:
    pdf_base64 = ctx.event.data.get("pdf_base64")
    source_id = ctx.event.data.get("source_id", "document.pdf")

    if not pdf_base64:
        raise ValueError("Missing 'pdf_base64' string context data.")

    # Write the incoming data safely to an internal OS ephemeral tempfile location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(base64.b64decode(pdf_base64))
        temp_path = temp_pdf.name

    try:
        chunks = load_and_chunk_pdf(temp_path)
    finally:
        # Erase temporary track traces to prevent garbage footprint retention leaks
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

def _search_step(question: str, top_k: int = 5) -> RAGSearchResult:
    query_vec = embed_texts([question])[0]
    store = QdrantStorage()
    found = store.search(query_vec, top_k)
    return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])



@inngest_client.create_function(
    fn_id="rag_ingest_pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
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
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search_step(question, top_k),
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

    # Inngest safely checkpoints and returns the string output from _call_llm
    answer = await ctx.step.run("llm-answer", _call_llm)

    return {
        "answer": answer.strip(),
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }


app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])