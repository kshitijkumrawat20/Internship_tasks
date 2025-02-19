# AI Chatbot with Multilingual Capabilities

This project is an AI chatbot application built using Streamlit and Google's Generative AI. The chatbot is designed to provide responses in the language of the user's input, making it a multilingual assistant.

## Features

- Multilingual support: Responds in the language of the user's input.
- Utilizes Google's Generative AI for generating responses.
- Maintains chat history for context-aware interactions.

## Requirements

To run this project, you need to have the following Python packages installed:

- streamlit
- transformers
- google-generativeai
- langchain_google_genai
- langchain
- langchain_community
- langchain_core
- python-dotenv

These dependencies are listed in the `requirements.txt` file.

## Setup Instructions

1. **Install the required packages:Use the following command to install the necessary Python packages:** 
    ```bash
    pip install -r requirements.txt

2. **Environment Variables:The application requires a .env file to store the Gemini API key. Create a .env file in the root directory of the project and add your Gemini API key:** 
    ```bash
    GEMINI_API_KEY=your_gemini_api_key_here


3. **Run the application:Start the Streamlit application by running:** 
    ```bash
    streamlit run app.py



