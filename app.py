import streamlit as st
from groq import Groq
import functions as f
import time

user_id = "user1"

# Initialize Groq client
client = Groq()

start_time = time.time()
# Load model and tokenizer once

if "models" not in st.session_state or "tokenizers" not in st.session_state:
    st.session_state.models, st.session_state.tokenizers = f.load_model_tokenizer()

print(f"\nload_model_tokenizer() : {time.time() - start_time}\n")

# Streamlit app setup
st.set_page_config(page_title="Property Preference Analysis", layout="wide")

# App header
st.title("💬 Chatroom for Prospect")
st.markdown("Welcome to the chat room! Start a conversation and find ur property preference.")

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

    if user_input.strip().lower() == "/bye":
        user_messages = [{'content': message['content']} for message in st.session_state.messages if message['role'] == 'user']
        messages = "\n".join([msg["content"] for msg in user_messages[:-1]])
        print(f"\nmessages : {messages}\n")

        traits = f.personality_analysis_sentence(messages, st.session_state.models, st.session_state.tokenizers)
        converted = f.convert_emojis(messages)
        validation_response = f.validate_personality_with_llm(converted, traits, client)
        property_recommendation = f.property_recommend(user_id, traits, client)
        df = f.analysis_result_output2(traits)
        sdf = f.analysis_result_output2(st.session_state.straits)

        st.session_state.messages = []
        with st.chat_message("assistant"):
            st.markdown(f"Thank you for chatting with me. Goodbye! Here are your results:")
            st.markdown("### Personality Traits Analysis - Previous Sentence:")
            st.dataframe(sdf, use_container_width=True)

            st.markdown("### Personality Traits Analysis - Complete Convo:")
            st.dataframe(df, use_container_width=True)

            st.markdown(f"### Property Recommendation:")
            st.markdown(f"{property_recommendation}")
            st.markdown(f"### Validation Response:")
            st.markdown(f"{validation_response}")
        st.stop()

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
        st.session_state.straits = f.personality_analysis_sentence(converted, st.session_state.models, st.session_state.tokenizers)
        analysis_time = time.time() - start_time

        print(f"\npersonality_analysis_sentence() : {analysis_time}\n")

        record = {"sender_id": user_id,"text": converted, "traits": st.session_state.straits}

        print(f"\nrecord : {record}\n")
                
        st.session_state.svalidation_response = f.validate_personality_with_llm(converted, st.session_state.straits, client)
        st.session_state.sproperty_recommendation = f.property_recommend(user_id, st.session_state.straits, client)
        
        for trait, details in st.session_state.straits.items():
            st.sidebar.write(f"**{trait.capitalize()}**: {details['value'].upper()} (Score: {details['score']})")
