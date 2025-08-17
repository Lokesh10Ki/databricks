import logging
import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv  
from databricks import sql
from databricks.sdk.core import Config
from langchain_community.llms import Databricks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"

# Helpers
def run_sql(query: str) -> pd.DataFrame:
    cfg = Config()  # uses DATABRICKS_HOST/TOKEN from env
    http_path = f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}"
    with sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            try:
                return cur.fetchall_arrow().to_pandas()
            except Exception:
                return pd.DataFrame()

def get_trips_schema_text() -> str:
    try:
        df = run_sql("DESCRIBE TABLE samples.nyctaxi.trips")
        df = df[df["col_name"].notna() & df["data_type"].notna()]
        cols = [f"{r.col_name} {r.data_type}" for _, r in df.iterrows()]
        return "Columns:\n- " + "\n- ".join(cols)
    except Exception as e:
        logger.warning(f"Could not fetch schema: {e}")
        # Fallback minimal schema
        return (
            "Columns:\n"
            "- trip_distance double\n- fare_amount double\n"
            "- pickup_zip int\n- dropoff_zip int\n- pickup_datetime timestamp\n- dropoff_datetime timestamp"
        )

def extract_sql(text: str) -> str:
    m = re.search(r"```sql(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip().rstrip(";")
    # fallback: take first line ending with FROM trips ...
    return text.strip().rstrip(";")

def generate_sql(question: str, schema_text: str, llm: Databricks) -> str:
    prompt = f"""
You are a Databricks SQL expert. Write a single SQL query for Databricks SQL to answer the user's question
using ONLY the table samples.nyctaxi.trips. Use the schema below. Return ONLY the SQL in a fenced ```sql block.

Schema:
{schema_text}

Guidelines:
- Prefer simple SELECT with WHERE/GROUP BY.
- Limit to at most 200 rows.
- Use fully qualified table name samples.nyctaxi.trips.
- If the question is ambiguous, make a reasonable assumption.

Question:
{question}
"""
    raw = llm.invoke(prompt)
    return extract_sql(raw)

def summarize_answer(question: str, df: pd.DataFrame, llm: Databricks) -> str:
    sample = df.head(10)
    csv_preview = sample.to_csv(index=False)
    prompt = f"""
You are a helpful analyst. Given the user's question and CSV preview of query results,
produce a concise answer. If data is insufficient, say so.

Question:
{question}

CSV preview (up to 10 rows):
{csv_preview}
"""
    return llm.invoke(prompt)

# App
missing = [k for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_WAREHOUSE_ID") if not os.getenv(k)]
if missing:
    st.warning(f"Missing env vars: {', '.join(missing)}")

llm = Databricks(endpoint_name=LLM_ENDPOINT_NAME)
schema_text = get_trips_schema_text()

st.title("🧱 NYCTaxi Q&A (samples.nyctaxi.trips)")
st.caption(f"Endpoint: {LLM_ENDPOINT_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about the taxi trips data..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            sql_text = generate_sql(question, schema_text, llm)
            st.markdown("Proposed SQL:")
            st.code(sql_text, language="sql")

            df = run_sql(f"{sql_text} LIMIT 200")
            if df.empty:
                answer = "No rows returned or query failed."
                st.warning(answer)
            else:
                st.dataframe(df, use_container_width=True)
                answer = summarize_answer(question, df, llm)
                st.markdown(answer)
        except Exception as e:
            answer = f"Error: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})