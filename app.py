import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq()

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

# predict here

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
    st.experimental_rerun()

# Instructions for running the app
st.sidebar.header("Instructions")
st.sidebar.markdown("1. Install the required packages with `pip install streamlit groq`.")
st.sidebar.markdown("2. Save this code to a file, e.g., `chat_app.py`.")
st.sidebar.markdown("3. Run the app using `streamlit run chat_app.py`.")
st.sidebar.markdown("4. Start chatting with LLaMA 3.3!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit and Groq API.")

# That’s it! Let me know if you want me to add anything. 🚀
