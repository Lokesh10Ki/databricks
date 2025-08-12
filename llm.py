import re
import logging
from typing import List
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

FOUNDATION_DEFAULTS = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-70b-instruct",
    "databricks-meta-llama-3-1-8b-instruct",
    "databricks-mixtral-8x7b-instruct",
]

def list_llm_endpoints() -> List[str]:
    try:
        w = WorkspaceClient(config=Config(auth_type="pat"))
        names = [e.name for e in w.serving_endpoints.list()]
        return sorted(set(names) | set(FOUNDATION_DEFAULTS))
    except Exception as e:
        logger.warning(f"Could not list serving endpoints: {e}")
        return FOUNDATION_DEFAULTS

def get_chat_llm(endpoint: str) -> ChatDatabricks:
    return ChatDatabricks(endpoint=endpoint)

def extract_sql(text: str) -> str:
    m = re.search(r"```sql(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip().rstrip(";")
    return text.strip().rstrip(";")

def first_statement(sql_text: str) -> str:
    parts = [p.strip() for p in re.split(r";\s*", sql_text) if p.strip()]
    return parts[0] if parts else sql_text

def ensure_limit(sql_text: str, limit: int = 200) -> str:
    return sql_text if re.search(r"\blimit\s+\d+\b", sql_text, flags=re.I) else f"{sql_text} LIMIT {limit}"

def generate_sql(question: str, schema_text: str, llm: ChatDatabricks) -> str:
    messages = [
        SystemMessage(content=(
            "You are a Databricks SQL expert. Write ONE single SQL statement for Databricks SQL. "
            "Use only samples.nyctaxi.trips. Return ONLY the SQL in a fenced ```sql block. "
            "Do NOT include multiple statements; exactly one SELECT."
        )),
        HumanMessage(content=f"Schema:\n{schema_text}\n\nQuestion:\n{question}")
    ]
    raw = llm.invoke(messages)
    return extract_sql(raw.content)

def refine_sql(question: str, schema_text: str, prev_sql: str, error_text: str, llm: ChatDatabricks) -> str:
    messages = [
        SystemMessage(content=(
            "You are a Databricks SQL expert. Fix the SQL given the error from Databricks. "
            "Return ONLY ONE valid SQL SELECT in a ```sql fenced block. No comments, no extra statements."
        )),
        HumanMessage(content=(
            f"Schema:\n{schema_text}\n\nQuestion:\n{question}\n\n"
            f"Previous SQL:\n{prev_sql}\n\nError from Databricks:\n{error_text}\n\n"
            "Provide the corrected SQL now."
        ))
    ]
    raw = llm.invoke(messages)
    return extract_sql(raw.content)

def summarize_answer(question: str, df: pd.DataFrame, llm: ChatDatabricks) -> str:
    sample = df.head(10)
    csv_preview = sample.to_csv(index=False)
    messages = [
        SystemMessage(content=(
            "You are a helpful analyst. Answer concisely based only on the provided CSV preview. "
            "If insufficient, say so."
        )),
        HumanMessage(content=f"Question:\n{question}\n\nCSV preview (up to 10 rows):\n{csv_preview}")
    ]
    resp = llm.invoke(messages)
    return resp.content