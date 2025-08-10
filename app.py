import logging
import os
import streamlit as st
# ...existing code...
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from langchain_community.llms import Databricks
# ...existing code...

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Remove WorkspaceClient and SERVING_ENDPOINT assertion
# w = WorkspaceClient()
# assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in resource yaml."

# Initialize LangChain Databricks LLM with your existing endpoint
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = Databricks(endpoint_name=LLM_ENDPOINT_NAME)

def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.title("🧱 Chatbot App")
st.write(f"Hi {user_info['user_name']}, this is a basic chatbot using your own serving endpoint")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        try:
            # Simple instruction prefix; include more context/history if desired
            assistant_response = llm.invoke(f"You are a helpful assistant.\nUser: {prompt}")
            st.markdown(assistant_response)
        except Exception as e:
            assistant_response = f"Error querying model: {e}"
            st.error(assistant_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})