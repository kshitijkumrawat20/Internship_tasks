import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import webbrowser
import datetime
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
import plotly.express as px

class ChatbotEngine:
    def __init__(self):
        self.setup_chatbot()
        
    def setup_chatbot(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open("intents.json").read())
        self.words = pickle.load(open("words.pkl", "rb"))
        self.classes = pickle.load(open("classes.pkl", "rb"))
        self.model = load_model("chatbotmodel.h5")

    def clean_up_sentence(self, sentence):
        return [self.lemmatizer.lemmatize(word.lower()) 
                for word in nltk.word_tokenize(sentence)]

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": self.classes[r[0]], "probability": str(r[1])} 
                for r in results]

    def get_response(self, intents_list):
        if not intents_list:
            return "I'm not sure how to respond to that. Can you please rephrase your question?"
        tag = intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm sorry, I don't have a specific response for that. Can you try asking something else?"

    def get_intent_category(self, intents_list):
        """Get the intent category from the prediction"""
        if not intents_list:
            return "unknown"
        return intents_list[0]["intent"]

def create_analytics_page():
    st.title("Chatbot Analytics Dashboard")
    
    # Add back button
    if st.button("← Back to Chat"):
        st.session_state.show_analytics = False
        st.rerun()
    
    # Create tabs for different analytics sections
    tab1, tab2, tab3, tab4 = st.tabs(["Usage Metrics", "Response Analysis", "User Satisfaction", "Topic Analysis"])
    
    with tab1:
        st.subheader("Usage Statistics")
        col1, col2, col3 = st.columns(3)
        
        # Basic metrics
        total_queries = st.session_state.analytics['total_queries']
        uptime = datetime.now() - st.session_state.analytics['start_time']
        avg_queries_per_hour = total_queries / (uptime.total_seconds() / 3600) if uptime.total_seconds() > 0 else 0
        
        col1.metric("Total Queries", total_queries)
        col2.metric("Uptime", f"{uptime.days}d {uptime.seconds//3600}h")
        col3.metric("Queries/Hour", f"{avg_queries_per_hour:.1f}")
        
        # Daily usage chart
        st.subheader("Daily Usage Trends")
        if st.session_state.analytics['daily_queries']:
            daily_data = pd.DataFrame(
                list(st.session_state.analytics['daily_queries'].items()),
                columns=['Date', 'Queries']
            )
            st.line_chart(daily_data.set_index('Date'))
            
        # Hourly distribution
        st.subheader("Query Distribution by Hour")
        if st.session_state.analytics['hourly_queries']:
            hourly_data = pd.DataFrame(
                list(st.session_state.analytics['hourly_queries'].items()),
                columns=['Hour', 'Count']
            )
            st.bar_chart(hourly_data.set_index('Hour'))

    with tab2:
        st.subheader("Response Time Analysis")
        
        if st.session_state.analytics['response_times']:
            response_times = st.session_state.analytics['response_times']
            avg_response = sum(response_times) / len(response_times)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Response Time", f"{avg_response:.3f}s")
            col2.metric("Fastest Response", f"{min(response_times):.3f}s")
            col3.metric("Slowest Response", f"{max(response_times):.3f}s")
            
            # Response time distribution
            st.subheader("Response Time Distribution")
            fig = px.histogram(response_times, nbins=20, title="Response Time Distribution")
            st.plotly_chart(fig)

    with tab3:
        st.subheader("User Satisfaction Metrics")
        
        if st.session_state.analytics['ratings']:
            ratings = st.session_state.analytics['ratings']
            positive_ratings = sum(1 for r in ratings if r == 1)
            total_ratings = len(ratings)
            satisfaction_rate = (positive_ratings / total_ratings) * 100 if total_ratings > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Ratings", total_ratings)
            col2.metric("Positive Ratings", positive_ratings)
            col3.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
            
            # Ratings pie chart
            fig = px.pie(
                values=[positive_ratings, total_ratings - positive_ratings],
                names=['Positive', 'Negative'],
                title="Rating Distribution"
            )
            st.plotly_chart(fig)
            
            # Ratings trend
            st.subheader("Satisfaction Trend")
            rolling_satisfaction = pd.Series(ratings).rolling(10).mean()
            st.line_chart(rolling_satisfaction)

    with tab4:
        st.subheader("Topic Analysis")
        
        # Top topics chart
        st.subheader("Most Common Topics")
        top_topics = st.session_state.analytics['topic_counter'].most_common(10)
        if top_topics:
            topic_data = pd.DataFrame(top_topics, columns=['Topic', 'Count'])
            fig = px.bar(topic_data, x='Topic', y='Count', title="Top 10 Topics")
            st.plotly_chart(fig)
            
            # Topic distribution pie chart
            fig = px.pie(topic_data, values='Count', names='Topic', title="Topic Distribution")
            st.plotly_chart(fig)

def initialize_analytics():
    """Initialize analytics in session state"""
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {
            'total_queries': 0,
            'daily_queries': {},
            'hourly_queries': {},
            'topic_counter': Counter(),
            'ratings': [],
            'response_times': [],
            'start_time': datetime.now()
        }
    if 'show_analytics' not in st.session_state:
        st.session_state.show_analytics = False

def update_analytics(intent_category, response_time):
    """Update analytics data"""
    today = datetime.now().date()
    current_hour = datetime.now().hour
    
    # Update total queries
    st.session_state.analytics['total_queries'] += 1
    
    # Update daily queries
    if today not in st.session_state.analytics['daily_queries']:
        st.session_state.analytics['daily_queries'][today] = 0
    st.session_state.analytics['daily_queries'][today] += 1
    
    # Update hourly queries
    if current_hour not in st.session_state.analytics['hourly_queries']:
        st.session_state.analytics['hourly_queries'][current_hour] = 0
    st.session_state.analytics['hourly_queries'][current_hour] += 1
    
    # Update topic counter
    st.session_state.analytics['topic_counter'][intent_category] += 1
    
    # Update response times
    st.session_state.analytics['response_times'].append(response_time)

def main():
    st.set_page_config(page_title="Enhanced NLP Chatbot", page_icon="🤖", layout="wide")
    
    # Initialize analytics
    initialize_analytics()
    
    if st.session_state.show_analytics:
        create_analytics_page()
    else:
        st.title("Task 1")
        
        # Initialize session states
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = ChatbotEngine()

        # Sidebar
        with st.sidebar:
            st.title("Options")
            
            if st.button("View Analytics"):
                st.session_state.show_analytics = True
                st.rerun()
            
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
            
            if st.button("Save Chat"):
                filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, "w") as f:
                    for message in st.session_state.chat_history:
                        f.write(f"{message['role'].title()}: {message['content']}\n")
                st.success(f"Chat history has been saved to {filename}")
            
            with st.expander("Help"):
                st.write("""
                Welcome to the Enhanced NLP Chatbot!

                Special Commands:
                - Type 'exit', 'quit', or 'bye' to end the conversation.
                - Type 'search <query>' to open a web search.
                - Type 'time' to get the current time.

                Features:
                - Clear Chat: Clears the current conversation.
                - Save Chat: Saves the conversation history to a file.
                - Help: Shows this help message.

                Enjoy chatting!
                """)
        
        # Main chat interface
        chat_col, _ = st.columns([3, 1])
        
        with chat_col:
            # Create a container for chat messages
            chat_container = st.container()
            
            # Process user input first
            user_message = st.chat_input("Type your message here...")
            
            if user_message:
                start_time = datetime.now()
                
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": user_message,
                    "id": f"{datetime.now().timestamp()}"
                })

                # Get chatbot response
                ints = st.session_state.chatbot.predict_class(user_message)
                bot_response = st.session_state.chatbot.get_response(ints)
                intent_category = st.session_state.chatbot.get_intent_category(ints)

                # Calculate response time
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Update analytics
                update_analytics(intent_category, response_time)

                # Add bot response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": bot_response,
                    "id": f"{datetime.now().timestamp()}"
                })
                
                # Rerun to update the chat display
                st.rerun()
            
            # Display chat history
            with chat_container:
                for idx, message in enumerate(st.session_state.chat_history):
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if message["role"] == "assistant":
                            message_id = message.get("id", f"{idx}_{datetime.now().timestamp()}")
                            cols = st.columns([1, 1, 1, 3])
                            if cols[0].button("👍", key=f"like_{message_id}"):
                                st.session_state.analytics['ratings'].append(1)
                                st.rerun()
                            if cols[1].button("👎", key=f"dislike_{message_id}"):
                                st.session_state.analytics['ratings'].append(0)
                                st.rerun()

if __name__ == "__main__":
    main()