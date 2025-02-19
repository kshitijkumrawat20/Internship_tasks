import streamlit as st
import base64
import io
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MultimodalChatbot:
    def __init__(self):
        """Initialize the chatbot with configuration and memory."""
        # API Key validation
        self.api_key = self._validate_api_key()
        
        # Initialize the multimodal model
        self.model = self._get_multimodal_model()
        
        # Conversation memory
        self.memory = ConversationBufferMemory()
        
        # Conversation chain
        self.conversation = ConversationChain(
            llm=self.model, 
            memory=self.memory,
            verbose=False
        )

    def _validate_api_key(self):
        """Validate and retrieve Google API key."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API Key is missing. Please set it in your .env file.")
            st.stop()
        return api_key

    def _get_multimodal_model(self):
        """Initialize the Gemini Pro Vision model."""
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.7,
                max_tokens=2048,
                api_key=self.api_key
            )
        except Exception as e:
            st.error(f"Error initializing model: {e}")
            st.stop()

    def process_image_input(self, uploaded_file):
        """Convert uploaded image to base64 for Gemini model."""
        if uploaded_file is not None:
            try:
                # Open the image
                image = Image.open(uploaded_file)
                
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()
                
                # Encode to base64
                return base64.b64encode(img_byte_arr).decode('utf-8')
            except Exception as e:
                st.error(f"Error processing image: {e}")
                return None
        return None

    def generate_response(self, messages):
        """Generate AI response with streaming support."""
        try:
            # Stream the response
            full_response = ""
            for chunk in self.model.stream(messages):
                full_response += chunk.content
                yield chunk.content
            
            # Add AI message to memory
            self.memory.chat_memory.add_ai_message(full_response)
        except Exception as e:
            st.error(f"Error generating response: {e}")
            yield f"An error occurred: {e}"

def main():
    # Page configuration
    st.set_page_config(
        page_title="MultiModal AI Chatbot", 
        page_icon="🤖",
        layout="wide"
    )

    # Initialize chatbot
    chatbot = MultimodalChatbot()

    # App title
    st.title("🤖 MultiModal AI Chatbot")
    st.write("Chat with an AI that understands both text and images!")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for image upload
    with st.sidebar:
        st.header("Image Input")
        uploaded_file = st.file_uploader(
            "Upload an image", 
            type=["png", "jpg", "jpeg"], 
            help="Upload an image to discuss"
        )

    # Chat input
    user_input = st.chat_input("Enter your message")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "image":
                st.image(message["content"])

    # Process user input
    if user_input or uploaded_file:
        # Prepare messages
        messages = []
        
        # Add image if uploaded
        if uploaded_file:
            base64_image = chatbot.process_image_input(uploaded_file)
            if base64_image:
                image_message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image in detail"},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ]
                )
                messages.append(image_message)
                
                # Display uploaded image
                st.session_state.messages.append({
                    "role": "user", 
                    "type": "image", 
                    "content": uploaded_file
                })

        # Add text input
        if user_input:
            messages.append(HumanMessage(content=user_input))
            st.session_state.messages.append({
                "role": "user", 
                "type": "text", 
                "content": user_input
            })

        # Get AI response
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in chatbot.generate_response(messages):
                full_response += chunk
                response_container.markdown(full_response)

            # Add AI response to session state
            st.session_state.messages.append({
                "role": "assistant", 
                "type": "text", 
                "content": full_response
            })

if __name__ == "__main__":
    main()