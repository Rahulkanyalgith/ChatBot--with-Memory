import streamlit as st
import os
from groq import Groq
from datetime import datetime
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Page configuration
st.set_page_config(
    page_title="Groq Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None

def get_custom_prompt():
    """Get custom prompt template based on selected persona"""
    persona = st.session_state.get('selected_persona', 'Default')
    
    personas = {
        'Default': """You are a helpful AI assistant.
                     Current conversation:
                     {history}
                     Human: {input}
                     AI:""",
        'Expert': """You are an expert consultant with deep knowledge across multiple fields.
                    Please provide detailed, technical responses when appropriate.
                    Current conversation:
                    {history}
                    Human: {input}
                    Expert:""",
        'Creative': """You are a creative and imaginative AI that thinks outside the box.
                      Feel free to use metaphors and analogies in your responses.
                      Current conversation:
                      {history}
                      Human: {input}
                      Creative AI:"""
    }
    
    return PromptTemplate(
        input_variables=["history", "input"],
        template=personas[persona]
    )

def main():
    initialize_session_state()
    
    # Sidebar Configuration
    with st.sidebar:
        st.title("🛠️ Chat Settings")
        
        # Model selection with custom styling
        st.subheader("Model Selection")
        model = st.selectbox(
            'Choose your model:',
            ['mixtral-8x7b-32768', 'llama2-70b-4096'],
            help="Select the AI model for your conversation"
        )
        
        # Memory configuration
        st.subheader("Memory Settings")
        memory_length = st.slider(
            'Conversation Memory (messages)',
            1, 10, 5,
            help="Number of previous messages to remember"
        )
        