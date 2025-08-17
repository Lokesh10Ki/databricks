import os
import logging
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from dbsql import run_sql, get_trips_schema_text
from llm import (
    list_llm_endpoints,
    get_chat_llm,
    generate_sql,
    refine_sql,
    summarize_answer,
    first_statement,
    ensure_limit,
)

from rag import ingest_uploaded_files, retrieve_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurable max retries for fixing broken SQL
MAX_SQL_RETRIES = int(os.getenv("MAX_SQL_RETRIES", "10"))

# Discover available LLM endpoints for dropdown
AVAILABLE_ENDPOINTS = list_llm_endpoints()
DEFAULT_ENDPOINT = AVAILABLE_ENDPOINTS[0] if AVAILABLE_ENDPOINTS else "databricks-meta-llama-3-3-70b-instruct"

# -------- App bootstrap --------
missing = [k for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_WAREHOUSE_ID") if not os.getenv(k)]
if missing:
    logger.warning(f"Missing env vars: {', '.join(missing)}")

schema_text = get_trips_schema_text()
try:
    tables_df = run_sql("SHOW TABLES IN samples.nyctaxi")
except Exception as e:
    logger.warning(f"Could not list tables: {e}")
    tables_df = pd.DataFrame()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "NYCTaxi Q&A"

def df_to_table(df: pd.DataFrame, max_rows: int = 30):
    if df is None or df.empty:
        return html.Div("No rows returned.", className="text-muted")
    cols = list(df.columns)
    rows = df.head(max_rows).to_dict("records")
    return html.Table([
        html.Thead(html.Tr([html.Th(c) for c in cols])),
        html.Tbody([html.Tr([html.Td(row.get(c)) for c in cols]) for row in rows])
    ], style={"width": "100%", "overflowX": "auto"})

app.layout = dbc.Container([
    html.H3("🧱 POC - Databricks AI Intelligence (samples.nyctaxi.trips)"),

    dbc.Row([
        dbc.Col([
            html.Label("Model endpoint"),
            dcc.Dropdown(
                id="endpoint-select",
                options=[{"label": name, "value": name} for name in AVAILABLE_ENDPOINTS],
                value=DEFAULT_ENDPOINT,
                clearable=False,
            ),
        ], width=6),
     dbc.Col([
            html.Label("Upload knowledge files (.txt/.md)"),
            dcc.Upload(
                id="file-upload",
                children=html.Div(["Drag and drop or click to upload"]),
                multiple=True,
                style={
                    "width": "100%", "height": "38px", "lineHeight": "38px",
                    "borderWidth": "1px", "borderStyle": "dashed",
                    "borderRadius": "6px", "textAlign": "center"
                },
            ),
            html.Small(id="upload-status", className="text-muted"),
        ], width=6),
    ], className="mb-3"),

    html.H6("Tables in samples.nyctaxi"),
    html.Div(id="tables-preview"),
    html.Hr(),

    dcc.Store(id="messages", data=[]),

    dbc.Row([
        dbc.Col([
            html.Div(id="chat-log", style={
                "height": "45vh", "overflowY": "auto", "border": "1px solid #ddd",
                "borderRadius": "6px", "padding": "8px", "backgroundColor": "white"
            })
        ], width=12)
    ], className="mb-2"),

    dbc.Row([
        dbc.Col(dbc.Input(id="user-input", type="text", placeholder="Ask about the taxi trips data..."), width=10),
        dbc.Col(dbc.Button("Send", id="send-btn", color="primary", n_clicks=0, style={"width": "100%"}), width=2),
    ], className="mb-3"),

    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div([
            html.H6("Proposed SQL"),
            html.Pre(id="sql-text", style={"whiteSpace": "pre-wrap"}),

            html.H6("Result preview"),
            html.Div(id="result-table"),

            html.H6("Answer"),
            dcc.Markdown(id="answer-text", link_target="_blank"),
        ])
    ),
], fluid=True)

@app.callback(
    Output("upload-status", "children"),
    Input("file-upload", "contents"),
    State("file-upload", "filename"),
    prevent_initial_call=True
)
def on_upload(contents, filenames):
    try:
        n = ingest_uploaded_files(contents or [], filenames or [])
        return f"Indexed {n} chunks from {len(filenames or [])} file(s)."
    except Exception as e:
        return f"Upload failed: {e}"

def render_chat(messages):
    children = []
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        bg = "#e9ecef" if m["role"] == "user" else "#f8f9fa"
        content = html.Div(m["content"]) if m["role"] == "user" else dcc.Markdown(m["content"], link_target="_blank")
        children.append(
            dbc.Card(dbc.CardBody([
                html.Small(who, className="text-muted"),
                content
            ]), style={"marginBottom": "8px", "backgroundColor": bg})
        )
    return children

@app.callback(
    Output("tables-preview", "children"),
    Input("endpoint-select", "value")
)
def on_load_tables(_):
    return df_to_table(tables_df, max_rows=100)

@app.callback(
    Output("chat-log", "children"),
    Output("messages", "data"),
    Output("sql-text", "children"),
    Output("result-table", "children"),
    Output("answer-text", "children"),
    Input("send-btn", "n_clicks"),
    State("user-input", "value"),
    State("messages", "data"),
    State("endpoint-select", "value"),
    prevent_initial_call=True
)
def on_send(n_clicks, user_text, messages, endpoint_name):
    messages = messages or []
    if not user_text:
        return render_chat(messages), messages, "", html.Div(), ""

    messages.append({"role": "user", "content": user_text})

    attempt_logs = []
    last_error = None
    df = pd.DataFrame()
    sql_final = ""

    chat_llm = get_chat_llm(endpoint_name)

    # Retrieve RAG context
    try:
        rag_context = retrieve_context(user_text, k=5)
    except Exception as _:
        rag_context = ""

    for attempt in range(1, MAX_SQL_RETRIES + 1):
        try:
            if attempt == 1:
                sql_candidate = generate_sql(user_text, schema_text, chat_llm, context=rag_context)
            else:
                sql_candidate = refine_sql(user_text, schema_text, sql_final, last_error or "Unknown error", chat_llm)

            sql_candidate = first_statement(sql_candidate)
            sql_candidate = ensure_limit(sql_candidate, 200)
            sql_final = sql_candidate
            attempt_logs.append(f"Attempt {attempt} SQL:\n{sql_candidate}")

            df = run_sql(sql_candidate)
            break
        except Exception as e:
            last_error = str(e)
            attempt_logs.append(f"Attempt {attempt} error:\n{last_error}")
            continue

    sql_text_out = "\n\n".join(attempt_logs)

    try:
        if df.empty:
            table = html.Div("No rows returned.", className="text-muted")
            answer = "No rows returned. Try refining your question."
        else:
            table = df_to_table(df)
            answer = summarize_answer(user_text, df, chat_llm, context=rag_context)
    except Exception as e:
        table = html.Div()
        answer = f"Error: {e}"

    messages.append({"role": "assistant", "content": answer})
    return render_chat(messages), messages, sql_text_out, table, answer

if __name__ == "__main__":
    app.run_server(debug=False, use_reloader=False)