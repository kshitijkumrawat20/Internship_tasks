# 🤖 AI Chatbot with Sentiment Analysis

## **📌 Project Overview**
This project implements an **AI-powered chatbot** that integrates **sentiment analysis** to detect user emotions and generate appropriate responses. The chatbot uses:
- **Streamlit** for the user interface.
- **DistilBERT** (from Hugging Face) for sentiment analysis.
- **Google Gemini (via LangChain)** for generating responses.
- **LangChain Prompt Templates** to structure responses based on sentiment.

### **🎯 Key Features**
✅ Real-time **sentiment detection** (Positive, Negative, Neutral).  
✅ **Context-aware chatbot responses** based on sentiment.  
✅ Uses **Google Gemini** via LangChain for dynamic AI-generated replies.  
✅ **Fast and lightweight UI** using Streamlit.  
✅ Easily **deployable** via Streamlit Cloud or local hosting.  

---

## **⚙️ Tech Stack**
| Component  | Technology Used  |
|------------|----------------|
| **Frontend** | Streamlit (Python-based UI) |
| **Sentiment Analysis** | DistilBERT (Hugging Face Transformers) |
| **AI Model** | Google Gemini (via LangChain) |
| **Backend Processing** | LangChain for LLM prompt generation |
| **Deployment** | Streamlit Cloud / Local Hosting |

---

## **📥 Installation & Setup**
### **1️⃣ Prerequisites**
Ensure you have **Python 3.8+** installed. Then, install the required dependencies:
```sh
pip install streamlit transformers google-generativeai langchain
```

### **2️⃣ Set Up Google Gemini API**
- Get an API key from [Google AI Studio](https://aistudio.google.com/).
- Create a `.env` file or directly configure the API key in the script:
```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

### **3️⃣ Run the Chatbot**
Execute the script using:
```sh
streamlit run app.py
```
This will launch the chatbot in your browser.

---

## **📝 Code Implementation**
### **1️⃣ Sentiment Analysis (DistilBERT)**
```python
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

user_input = "I'm feeling really frustrated with this service."
result = sentiment_analyzer(user_input)[0]
sentiment = result['label']  # Outputs: POSITIVE, NEGATIVE, or NEUTRAL
```

### **2️⃣ LangChain + Google Gemini for AI Response**
```python
from langchain.llms import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

genai.configure(api_key="YOUR_GEMINI_API_KEY")
llm = GoogleGenerativeAI(model="gemini-pro")

prompt_template = PromptTemplate(
    template="User Message: {user_input}\nSentiment: {sentiment}\nRespond appropriately considering the sentiment.",
    input_variables=["user_input", "sentiment"],
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)
response = llm_chain.run({"user_input": user_input, "sentiment": sentiment})
```

### **3️⃣ Streamlit UI Integration**
```python
import streamlit as st

st.title("🤖 AI Chatbot with Sentiment Analysis")
user_input = st.text_input("Enter your message:")
if user_input:
    sentiment = sentiment_analyzer(user_input)[0]["label"]
    response = llm_chain.run({"user_input": user_input, "sentiment": sentiment})
    st.subheader("Sentiment Analysis Result")
    st.write(f"**Detected Sentiment:** {sentiment}")
    st.subheader("Chatbot Response")
    st.write(response)
```

---

## **🚀 Deployment**
### **Option 1: Run Locally**
```sh
streamlit run app.py
```

### **Option 2: Deploy on Streamlit Cloud**
1. **Push code to GitHub.**
2. Go to [Streamlit Cloud](https://share.streamlit.io/), connect GitHub repo.
3. Deploy the Streamlit app!

---

## **📊 Evaluation Criteria**
| Criteria | Description |
|----------|-------------|
| **Accuracy** | Correctly classifies sentiment (Positive, Negative, Neutral) |
| **Response Appropriateness** | AI-generated responses align with user sentiment |
| **User Experience** | Smooth and intuitive chatbot interaction |
| **Performance** | Fast response time with minimal latency |

---

## **🛠️ Future Enhancements**
🔹 **Multilingual Sentiment Analysis** using XLM-RoBERTa.  
🔹 **Emotion Recognition** (Happy, Sad, Angry) instead of just 3-class sentiment.  
🔹 **Chat History Persistence** for better context tracking.  
🔹 **Fine-tuning Gemini Responses** for domain-specific chatbots.  

---

## **📌 Conclusion**
This AI-powered chatbot successfully integrates **sentiment analysis** to improve user interactions. By detecting sentiment and tailoring responses dynamically, it enhances user experience and **bridges the gap between AI and emotional intelligence**. 🚀

