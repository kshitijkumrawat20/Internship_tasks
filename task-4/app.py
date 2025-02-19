import streamlit as st
from transformers import pipeline
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize Google Gemini with API Key
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

# Sentiment Analysis Pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Prompt Template for Chatbot Responses
prompt_template = PromptTemplate(
    input_variables=["user_input", "sentiment"],
    template="User said: {user_input}\nSentiment detected: {sentiment}\nResponse:"
)

llm_chain =prompt_template | llm

# Streamlit UI
st.set_page_config(page_title="Task-4", page_icon='🤖')
st.title("🤖 AI Chatbot with Sentiment Analysis")
st.write("Chatbot detects sentiment and responds accordingly.")

# User Input
user_input = st.text_input("Enter your message:")

if user_input:
    # Analyze Sentiment
    result = sentiment_analyzer(user_input)[0]
    sentiment = result["label"]  # POSITIVE, NEGATIVE, or NEUTRAL

    # Pass input as a dictionary (not formatted string)
    response = llm_chain.invoke({"user_input": user_input, "sentiment": sentiment})

    # Display Results
    st.subheader("Sentiment Analysis Result")
    st.write(f"**Detected Sentiment:** {sentiment}")

    st.subheader("Chatbot Response")
    st.write(response.content)