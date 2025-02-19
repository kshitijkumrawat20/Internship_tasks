import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Google Gemini with API Key
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

# Prompt Template for Chatbot Responses
prompt_template = PromptTemplate(
    input_variables=["user_input", "chat_history"],
    template="You are a multilingual assitant please provide the answer in the laguage in which the user provided the input -   Chat History: {chat_history}\nUser: {user_input}\nResponse:"
)

llm_chain = prompt_template | llm

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon='🤖')
st.title("🤖 AI Chatbot with Multilingual capabilities.")

# Initialize session state for chat history
if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []

# Display chat messages from history
for message in st.session_state.chat_history:
    if isinstance(message, tuple) and len(message) == 2:
        role, text = message
        with st.chat_message(role):
            st.markdown(text)

# User Input
user_input = st.chat_input("Enter your message:")

if user_input:
    # Append user input to chat history
    st.session_state.chat_history.append(("user", user_input))
    
    # Get response from LLM
    response = llm_chain.invoke({
        "user_input": user_input,
        "chat_history": '\n'.join([f'{role}: {text}' for role, text in st.session_state.chat_history if isinstance((role, text), tuple) and len((role, text)) == 2])
    })
    
    # Append chatbot response to chat history
    st.session_state.chat_history.append(("assistant", response.content))
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(response.content)
