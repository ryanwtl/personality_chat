import streamlit as st
import pandas as pd
from functions import get_llama_response, personality_analysis_sentence, load_model_tokenizer, display_radar_chart, property_prediction

# Load model and tokenizer once
if "models" not in st.session_state or "tokenizers" not in st.session_state:
    st.session_state.models, st.session_state.tokenizers = load_model_tokenizer()

# Streamlit UI setup
st.set_page_config(page_title="Business Chatroom", layout="wide")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Business Chatroom"])

if page == "Home":
    st.title("Tenant EDA")
    st.write("### Objectives for Prospect Tenant")
    st.markdown("""
    1. **Personality Traits Monitoring and Validity Check**
    2. **Determine an individual's property preferences through chat interactions**
    3. **Discover potential prospects to become tenants**
    """)
    
    st.write("### Evaluation with Belive Dataset")
    
    # Load and display CSV for Length Analysis
    st.subheader("Performance Based on Input Sentence Length")
    try:
        df_length = pd.read_csv("Y:/TARUMT/Project/Belive Personality Chatroom/results/table1.csv")
        st.dataframe(df_length)
    except Exception as e:
        st.error(f"Error loading length data: {e}")
    
    # Load and display CSV for Language Analysis
    st.subheader("Performance Based on Input Sentence Language")
    try:
        df_language = pd.read_csv("Y:/TARUMT/Project/Belive Personality Chatroom/results/table2.csv")
        st.dataframe(df_language)
    except Exception as e:
        st.error(f"Error loading language data: {e}")
    
    # Display Personality Traits Image
    st.subheader("Personality Traits Among Belive Tenants")
    try:
        st.image(r"Y:/TARUMT/Project/Belive Personality Chatroom/results/Personality Traits among Tenants from Belive.png", caption="Distribution of BIG Five Traits")
    except Exception as e:
        st.error(f"Error loading personality traits image: {e}")
    
    # Display Number of Tenants and Traits Image
    st.subheader("Number of Tenants and Their Traits")
    try:
        st.image(r"Y:\TARUMT\Project\Belive Personality Chatroom\results\Number of Tenants vs. Co-occurring Traits.png", caption="Number of tenants categorized by personality traits")
    except Exception as e:
        st.error(f"Error loading tenant traits image: {e}")

else:
    # Streamlit UI setup
    st.title("📊 Business Chatroom with Personality Monitoring")
    st.sidebar.header("Chat Settings")

    with st.sidebar:
        threshold = st.slider("Set Threshold Value", 0.0, 1.0, 0.6, 0.01)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "personality_scores" not in st.session_state:
        st.session_state["personality_scores"] = []

    # Display chat history
    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "user" and i < len(st.session_state["personality_scores"]):
                personality_results = st.session_state["personality_scores"][i]
                
                # Create radar chart
                with st.expander("Personality Analysis", expanded=False):
                    print(f"\ntraits_{i}_radar_chart:{personality_results}")
                    if personality_results:
                        display_radar_chart(personality_results,threshold)

    # Chat input
    user_input = st.chat_input("Type your message...")
    if user_input:
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Check for exit command
        if user_input.strip().lower() == "/bye":
            messages_analyze = "/n".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])
            st.session_state["messages"].append({"role": "assistant", "content": "Thank you. Byeee"})
            with st.chat_message("assistant"):
                st.markdown("Thank you. Byeee")
                st.markdown("Proceed to Property Analysis with personality traits result")
                st.markdown("Please wait...")
            
            personality_results = personality_analysis_sentence(messages_analyze, st.session_state.models, st.session_state.tokenizers, threshold)
            property_prediction(personality_results,threshold)
            
            # st.stop()
            
        else:
            # Personality analysis
            personality_results = personality_analysis_sentence(user_input, st.session_state.models, st.session_state.tokenizers, threshold)
            st.session_state["personality_scores"].append(personality_results)
            
            with st.expander("Personality Analysis", expanded=False):
                if personality_results:
                    print(f"\ncurrent_traits_radar_chart:\n{personality_results}")
                    # Create radar chart
                    display_radar_chart(personality_results,threshold)
            
            # Get AI response
            ai_response = get_llama_response(user_input)
            st.session_state["messages"].append({"role": "assistant", "content": ai_response})
            st.session_state["personality_scores"].append(0)
            with st.chat_message("assistant"):
                st.markdown(ai_response)
