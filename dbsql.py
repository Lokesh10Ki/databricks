import os
import atexit
import logging
import pandas as pd
from databricks import sql
from databricks.sdk.core import Config

logger = logging.getLogger(__name__)

# Reuse a single SQL connection
os.environ.setdefault("DATABRICKS_AUTH_TYPE", "pat")
_cfg = Config(auth_type="pat")  # uses DATABRICKS_HOST/TOKEN
_HTTP_PATH = f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}"
_SQL_CONN = None

def _get_conn():
    global _SQL_CONN
    if _SQL_CONN is not None:
        return _SQL_CONN
    if not _cfg.host or not _HTTP_PATH:
        raise RuntimeError("DATABRICKS_HOST and DATABRICKS_WAREHOUSE_ID must be set.")
    _SQL_CONN = sql.connect(
        server_hostname=_cfg.host,
        http_path=_HTTP_PATH,
        credentials_provider=lambda: _cfg.authenticate
    )
    return _SQL_CONN

def _close_conn():
    global _SQL_CONN
    try:
        if _SQL_CONN is not None:
            _SQL_CONN.close()
    finally:
        _SQL_CONN = None

atexit.register(_close_conn)

def run_sql(query: str) -> pd.DataFrame:
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(query)
            try:
                return cur.fetchall_arrow().to_pandas()
            except Exception:
                return pd.DataFrame()
    except Exception as e:
        logger.warning(f"SQL error (first attempt): {e}")
        _close_conn()
        conn = _get_conn()
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
        return (
            "Columns:\n"
            "- trip_distance double\n- fare_amount double\n"
            "- pickup_zip int\n- dropoff_zip int\n- pickup_datetime timestamp\n- dropoff_datetime timestamp"
        )