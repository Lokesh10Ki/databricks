import os
from time import sleep
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

load_dotenv()

VS_ENDPOINT = os.getenv("RAG_VS_ENDPOINT", "rag-endpoint")
INDEX_NAME = os.getenv("RAG_VS_INDEX", "workspace.rag.docs_index")
EMBEDDING_EP = os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")

# NEW: read and validate creds, pass to SDK
host = (os.getenv("DATABRICKS_HOST") or "https://dbc-51a96bc5-edf3.cloud.databricks.com/").rstrip("/")
token = os.getenv("DATABRICKS_TOKEN") or "dapic27773b6e85bb754f627f930f64cf9fa"


if not host or not token:
    raise RuntimeError("Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env or env.")

w = WorkspaceClient(config=Config(host=host, token=token))

# Ensure endpoint
endpoints = {e.name: e for e in w.vector_search_endpoints.list_endpoints()}
if VS_ENDPOINT not in endpoints:
    print(f"Creating Vector Search endpoint: {VS_ENDPOINT}")
    w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT)
else:
    print(f"Endpoint exists: {VS_ENDPOINT}")

def is_ready(ep_obj) -> bool:
    # Handle various SDK response shapes
    st = getattr(ep_obj, "state", None)
    # dict-like
    if isinstance(st, dict):
        if st.get("ready") is True:
            return True
        detailed = str(st.get("status") or st.get("detailed_state") or "").upper()
        return detailed in ("READY", "ONLINE", "RUNNING")
    # object-like
    ready_attr = getattr(st, "ready", None)
    if ready_attr is True:
        return True
    detailed = str(getattr(st, "status", "") or getattr(st, "detailed_state", "")).upper()
    return detailed in ("READY", "ONLINE", "RUNNING")

# Wait for endpoint to be ready (with timeout)
max_tries = 60  # ~5 minutes
for i in range(max_tries):
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
    print(f"Endpoint state: {getattr(ep, 'state', None)}")
    if is_ready(ep):
        print("Endpoint is READY.")
        break
    sleep(5)
else:
    print("Warning: endpoint did not report READY within timeout, continuing...")

# Ensure direct index
indexes = {i.name: i for i in w.vector_search_indexes.list_indexes(endpoint_name=VS_ENDPOINT)}
if INDEX_NAME not in indexes:
    print(f"Creating direct index: {INDEX_NAME}")
    w.vector_search_indexes.create_direct_index(
        endpoint_name=VS_ENDPOINT,
        name=INDEX_NAME,
        primary_key="id",
        embedding_vector_columns=[{
            "name": "embedding",
            "embedding_model_endpoint_name": EMBEDDING_EP,
            "source_column": "content",
        }],
    )
else:
    print(f"Index exists: {INDEX_NAME}")

print("Done.")