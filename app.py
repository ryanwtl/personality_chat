import streamlit as st
from groq import Groq
import functions as f
import requests
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Initialize Groq client
client = Groq()

start_time = time.time()
# Load model and tokenizer once
if "models" not in st.session_state or "tokenizer" not in st.session_state:
    st.session_state.models, st.session_state.tokenizers = f.load_model_tokenizer(huggingface_api_key)

print(f"\nload_model_tokenizer() : {time.time() - start_time}\n")

# Streamlit app setup
st.set_page_config(page_title="LLaMA 3.3 Chat Room", layout="wide")

# App header
st.title("💬 Chat with LLaMA 3.3")
st.markdown("Welcome to the chat room! Start a conversation with LLaMA 3.3 below.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream LLaMA response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=st.session_state.messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            response_placeholder.markdown(full_response)
        
        # Save assistant message to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a button to clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Personality trait prediction
st.sidebar.header("Personality Traits")
if st.session_state.messages:
    latest_user_message = next((msg['content'] for msg in reversed(st.session_state.messages) if msg['role'] == 'user'), "")
    if latest_user_message:
        converted = f.convert_emojis(latest_user_message)

        start_time = time.time()
        traits = f.personality_analysis_sentence(converted, st.session_state.models, st.session_state.tokenizers)
        analysis_time = time.time() - start_time

        print(f"\npersonality_analysis_sentence() : {analysis_time}\n")

        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        record = {"text": converted, "traits": traits}

        print(f"\nrecord : {record}\n")
        print(f"\nrecord_type : {type(record)}\n")

        data = json.dumps(record)

        try:
            response = requests.post("http://localhost:8000", json=data, headers=headers, timeout=5)
            response.raise_for_status()  # Raise exception for HTTP errors
            print(f"\nresponse : {response.text}\n")
        except requests.exceptions.ConnectionError:
            print("Receiver is not running. Please start the receiver and try again.")
        except requests.exceptions.Timeout:
            print("Request timed out. Please check the receiver status.")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code} - {e.response.reason}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        
        for trait, details in traits.items():
            st.sidebar.write(f"**{trait.capitalize()}**: {details['value'].upper()} (Score: {details['score']})")
