import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv()


qdrant_url = os.getenv("QDRANT_URL", os.getenv("QDRANT_HOST_URL"))
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if "io:6333" in qdrant_url and qdrant_url.startswith("https://"):
    qdrant_url = qdrant_url.replace("https://", "")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

print("Connecting to Qdrant to apply index fix...")
client.create_payload_index(
    collection_name="docs",
    field_name="source",
    field_schema=PayloadSchemaType.KEYWORD
)
print("Successfully created metadata filter index!")