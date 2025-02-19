import streamlit as st
import faiss 
import pickle
from sentence_transformers import SentenceTransformer

# Load FAISS index and QA mapping
@st.cache_resource
def load_resources():
    # Load FAISS index
    index = faiss.read_index("faiss_index.bin")
    # Load QA mapping
    with open("qa_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')    
    return index, mapping, model

faiss_index, qa_mapping, model = load_resources()

# Retrieval function
def retrieve_answer(user_query):
    query_embedding = model.encode([user_query])
    _, indices = faiss_index.search(query_embedding, k=1)
    closest_idx = indices[0][0]
    return qa_mapping[closest_idx]

# Streamlit app
st.title("Medical Q&A Chatbot")
st.write("Ask your medical questions and get relevant answers from the MedQuAD dataset.")

user_query = st.text_input("Enter your medical question:")
if user_query:
    result = retrieve_answer(user_query)
    st.write(f"**Question:** {result['question']}")
    st.write(f"**Answer:** {result['answer']}")
    st.write(f"**Focus:** {result['focus']}")
    st.write(f"**Question Type:** {result['question_type']}")    
