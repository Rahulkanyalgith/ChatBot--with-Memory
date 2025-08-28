import streamlit as st
import os
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

# --- Custom CSS for UI polish ---
st.markdown("""
    <style>
    /* Main Title */
    h1 {text-align: center; font-size: 2.2rem; margin-bottom: 15px; color: #333;}
    
    /* Sidebar */
    .css-1d391kg {background: #f7f9fc;}
    .sidebar .sidebar-content {padding: 20px;}
    
    /* Chat Bubbles */
    .user-msg {background-color: #DCF8C6; padding: 12px; border-radius: 12px; margin: 5px 0; text-align: right;}
    .ai-msg {background-color: #F1F0F0; padding: 12px; border-radius: 12px; margin: 5px 0; text-align: left;}
    
    /* Metrics */
    .stMetric {background: #f5f5f5; padding: 10px; border-radius: 8px;}
    
    /* Buttons */
    div.stButton > button {border-radius: 10px; font-weight: 600;}
    </style>
""", unsafe_allow_html=True)

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
        st.title("⚙️ Chat Settings")
        
        with st.expander("🤖 Model Selection", expanded=True):
            model = st.selectbox(
                'Choose your model:',
                ['deepseek-r1-distill-llama-70b', 'meta-llama/llama-guard-4-12b'],
                help="Select the AI model for your conversation"
            )
        
        with st.expander("🧠 Memory Settings", expanded=True):
            memory_length = st.slider(
                'Conversation Memory (messages)',
                1, 30, 10,
                help="Number of previous messages to remember"
            )
        
        with st.expander("🎭 AI Persona", expanded=True):
            st.session_state.selected_persona = st.selectbox(
                'Select conversation style:',
                ['Default', 'Expert', 'Creative']
            )
        
        # Chat statistics
        if st.session_state.start_time:
            st.markdown("### 📊 Chat Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", len(st.session_state.chat_history))
            with col2:
                duration = datetime.now() - st.session_state.start_time
                st.metric("Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.start_time = None
            st.rerun()

    # --- Main chat interface ---
    st.title("🤖 Groq Chat Assistant")

    # Initialize chat components
    memory = ConversationBufferWindowMemory(k=memory_length)
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    conversation = ConversationChain(llm=groq_chat, memory=memory, prompt=get_custom_prompt())

    # Load chat history into memory
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Display chat history with custom bubbles
    for message in st.session_state.chat_history:
        st.markdown(f"<div class='user-msg'>👤 {message['human']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-msg'>🤖 {message['AI']}</div>", unsafe_allow_html=True)

    # User input section
    st.markdown("### 💭 Your Message")
    user_question = st.text_area(
        "",
        height=100,
        placeholder="Type your message here... (Shift + Enter to send)",
        key="user_input"
    )

    # Input buttons
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        send_button = st.button("📤 Send", use_container_width=True)
    with col3:
        if st.button("🔄 New Topic", use_container_width=True):
            memory.clear()
            st.success("Memory cleared for new topic!")

    if send_button and user_question:
        if not st.session_state.start_time:
            st.session_state.start_time = datetime.now()

        with st.spinner('🤔 Thinking...'):
            try:
                response = conversation(user_question)
                message = {'human': user_question, 'AI': response['response']}
                st.session_state.chat_history.append(message)
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        f"🌐 Powered by **Groq AI + LangChain** | Persona: *{st.session_state.selected_persona}* | Memory: *{memory_length} messages*"
    )

if __name__ == "__main__":
    main()
