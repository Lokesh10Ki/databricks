import base64
import logging
import os
from typing import List, Tuple

from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

logger = logging.getLogger(__name__)

def _get_embeddings() -> DatabricksEmbeddings:
    endpoint = os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
    return DatabricksEmbeddings(endpoint=endpoint)

def _get_vs() -> DatabricksVectorSearch:
    endpoint = os.getenv("RAG_VS_ENDPOINT", "rag-endpoint")
    index = os.getenv("RAG_VS_INDEX", "workspace.rag.docs_index")
    return DatabricksVectorSearch(
        endpoint=endpoint,
        index_name=index,
        embedding=_get_embeddings(),
    )

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks

def _decode_content(content_b64: str) -> str:
    # dcc.Upload provides content like "data:<mime>;base64,<payload>"
    _, b64data = content_b64.split(",", 1)
    return base64.b64decode(b64data).decode("utf-8", errors="ignore")

def ingest_uploaded_files(contents: List[str], filenames: List[str]) -> int:
    """Return number of chunks indexed."""
    if not contents:
        return 0
    vs = _get_vs()
    total_chunks = 0
    for content_b64, fname in zip(contents, filenames or []):
        try:
            text = _decode_content(content_b64)
        except Exception as e:
            logger.warning(f"Failed to decode {fname}: {e}")
            continue
        # Basic support for .txt/.md; extend as needed for PDF/Docx
        chunks = _chunk_text(text)
        if not chunks:
            continue
        metadatas = [{"source": fname}] * len(chunks)
        vs.add_texts(texts=chunks, metadatas=metadatas)
        total_chunks += len(chunks)
    return total_chunks

def retrieve_context(query: str, k: int = 5) -> str:
    vs = _get_vs()
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)