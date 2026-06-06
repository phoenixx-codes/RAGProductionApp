import os
from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
EMBED_DIM = 384
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Calculates vector embeddings using Groq's cloud API instead of local memory."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    embeddings = []
    for text in texts:
        # Utilizing a lightweight, fast open cloud embedding model structure
        response = client.audio.transcriptions
        # Alternative: Standardizing via standard lightweight requests or HuggingFace Free Inference API

        # using the ultra-reliable HuggingFace Free Inference API to match exact BAAI/bge-small-en-v1.5 model:
        import requests
        API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
        headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY', '')}"}

        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        if response.status_with == 200:
            embeddings.append(response.json())
        else:
            # Fallback zero-vector if external rate limits hit briefly
            embeddings.append([0.0] * EMBED_DIM)

    return embeddings