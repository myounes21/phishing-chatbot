"""
Phishing Campaign Chatbot
Main application with Streamlit UI
"""

import streamlit as st
import os
import sys
from typing import Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from data_processor import PhishingDataProcessor
from insight_generator import InsightGenerator
from embeddings import EmbeddingGenerator
from vector_store import VectorStore, RAGRetriever
from llm_orchestrator import LLMOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhishingChatbot:
    """
    Main chatbot application
    """
    
    def __init__(self):
        """Initialize the chatbot"""
        self.data_processor = None
        self.insight_generator = None
        self.embedding_generator = None
        self.vector_store = None
        self.rag_retriever = None
        self.orchestrator = None
        self.initialized = False
    
    def initialize(self, csv_path: str, groq_api_key: str):
        """
        Initialize all components
        
        Args:
            csv_path: Path to CSV file
            groq_api_key: Groq API key
        """
        try:
            with st.spinner("Initializing system components..."):
                # Step 1: Load and process data
                st.write("📊 Loading phishing campaign data...")
                self.data_processor = PhishingDataProcessor(csv_path)
                self.data_processor.load_data()
                
                # Step 2: Generate insights
                st.write("🧠 Generating analytical insights...")
                self.insight_generator = InsightGenerator(self.data_processor)
                insights = self.insight_generator.generate_all_insights()
                
                # Step 3: Create embeddings
                st.write("🔢 Creating text embeddings...")
                self.embedding_generator = EmbeddingGenerator()
                insights_with_embeddings = self.embedding_generator.encode_insights(insights)
                
                # Step 4: Setup vector store
                st.write("💾 Setting up vector database...")
                self.vector_store = VectorStore()
                self.vector_store.connect()
                self.vector_store.create_collection(recreate=True)
                self.vector_store.add_insights(insights_with_embeddings)
                
                # Step 5: Create RAG retriever
                st.write("🔍 Configuring RAG system...")
                self.rag_retriever = RAGRetriever(
                    self.vector_store,
                    self.embedding_generator
                )
                
                # Step 6: Initialize LLM orchestrator
                st.write("🤖 Connecting to Groq LLM...")
                self.orchestrator = LLMOrchestrator(
                    api_key=groq_api_key,
                    data_processor=self.data_processor,
                    rag_retriever=self.rag_retriever
                )
                
                self.initialized = True
                st.success("✅ System initialized successfully!")
                
                # Show summary statistics
                self._show_data_summary()
                
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            st.error(f"Initialization failed: {str(e)}")
            raise
    
    def _show_data_summary(self):
        """Display summary statistics"""
        if not self.data_processor:
            return
        
        df = self.data_processor.df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Emails", len(df))
        
        with col2:
            st.metric("Unique Users", df['User_ID'].nunique())
        
        with col3:
            clicked = len(df[df['Action'] == 'Clicked'])
            click_rate = (clicked / len(df)) * 100
            st.metric("Overall Click Rate", f"{click_rate:.1f}%")
        
        with col4:
            reported = len(df[df['Action'] == 'Reported'])
            report_rate = (reported / len(df)) * 100
            st.metric("Report Rate", f"{report_rate:.1f}%")
    
    def query(self, user_query: str) -> str:
        """
        Process user query
        
        Args:
            user_query: User's question
            
        Returns:
            Response string
        """
        if not self.initialized:
            return "System not initialized. Please upload data and initialize first."
        
        try:
            response = self.orchestrator.process_query(user_query)
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error: {str(e)}"


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Phishing Campaign Analyzer",
        page_icon="🎣",
        layout="wide"
    )
    
    st.title("🎣 Intelligent Phishing Campaign Analyzer")
    st.markdown("*Transform complex phishing simulation data into actionable insights*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if API key exists in environment
        env_api_key = os.getenv('GROQ_API_KEY')
        
        if env_api_key:
            st.success("✅ API key loaded from .env file")
            groq_api_key = env_api_key
            # Show masked version
            masked_key = env_api_key[:8] + "..." + env_api_key[-4:] if len(env_api_key) > 12 else "***"
            st.caption(f"Using key: `{masked_key}`")
            
            # Option to override
            with st.expander("🔧 Override API key (optional)"):
                override_key = st.text_input(
                    "Enter different API key",
                    type="password",
                    help="Leave empty to use .env key"
                )
                if override_key:
                    groq_api_key = override_key
                    st.info("Using provided key instead of .env")
        else:
            st.warning("⚠️ No API key found in .env file")
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Enter your Groq API key or set GROQ_API_KEY in .env file"
            )
            if groq_api_key:
                st.info("💡 Tip: Add to .env file to skip this step next time")
        
        # File upload
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload phishing campaign results CSV"
        )
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            if not groq_api_key:
                st.error("Please provide Groq API key (in .env or above)")
            elif not uploaded_file:
                st.error("Please upload a CSV file")
            else:
                # Save uploaded file temporarily
                with open("temp_upload.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize chatbot
                if 'chatbot' not in st.session_state:
                    st.session_state.chatbot = PhishingChatbot()
                
                st.session_state.chatbot.initialize("temp_upload.csv", groq_api_key)
        
        st.divider()
        
        # Example queries
        st.header("💡 Example Queries")
        example_queries = [
            "What is the click rate for Finance?",
            "Which department is most vulnerable?",
            "Why did the urgent template work well?",
            "Who are the top 5 riskiest users?",
            "What templates are most effective?",
            "How fast do people respond?",
            "Give me a summary of the campaign"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query}"):
                st.session_state.current_query = query
    
    # Main chat interface
    if 'chatbot' in st.session_state and st.session_state.chatbot.initialized:
        st.header("💬 Chat Interface")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle example query button clicks
        if 'current_query' in st.session_state:
            user_input = st.session_state.current_query
            del st.session_state.current_query
        else:
            user_input = st.chat_input("Ask about your phishing campaign...")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = st.session_state.chatbot.query(user_input)
                    st.markdown(response)
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            if hasattr(st.session_state.chatbot, 'orchestrator'):
                st.session_state.chatbot.orchestrator.clear_history()
            st.rerun()
    
    else:
        # Welcome screen
        st.info("👈 Please configure and initialize the system using the sidebar")
        
        # Check if .env exists
        if os.path.exists('.env'):
            st.success("✅ Found .env file - API key will be loaded automatically!")
        else:
            st.warning("⚠️ No .env file found. You can:")
            st.code("# Create .env file\necho GROQ_API_KEY=your_key_here > .env", language="bash")
            st.caption("Or enter API key manually in the sidebar")
        
        st.header("🎯 Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quantitative Analysis")
            st.markdown("""
            - Click rates by department
            - Template effectiveness metrics
            - User risk scoring
            - Response time analysis
            - Statistical insights
            """)
        
        with col2:
            st.subheader("Qualitative Insights")
            st.markdown("""
            - Behavioral pattern analysis
            - Psychological trigger identification
            - Contextual explanations
            - Actionable recommendations
            - Risk factor analysis
            """)
        
        st.header("📊 How It Works")
        st.markdown("""
        1. **Upload Data**: CSV file with phishing campaign results
        2. **Processing**: Automatically analyzes data using Pandas
        3. **Insight Generation**: Creates qualitative insights from patterns
        4. **Vector Storage**: Stores insights in Qdrant for semantic search
        5. **Intelligent Query**: Ask questions in natural language
        6. **Smart Response**: Groq LLM orchestrates tools to provide answers
        """)
        
        st.header("📋 CSV Format")
        st.markdown("""
        Your CSV should contain these columns:
        - `User_ID`: Unique identifier for each user
        - `Department`: User's department
        - `Template`: Phishing template used
        - `Action`: User action (Clicked/Ignored/Reported)
        - `Response_Time_Sec`: Time taken to respond (seconds)
        """)
        
        # Show sample data
        with st.expander("View Sample Data Format"):
            st.code("""User_ID,Department,Template,Action,Response_Time_Sec
U001,Finance,Urgent Password Reset,Clicked,35
U002,Sales,CEO Impersonation,Ignored,0
U003,IT,Fake Invoice,Reported,120""")


if __name__ == "__main__":
    main()