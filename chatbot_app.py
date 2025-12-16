"""
Phishing Campaign Chatbot
Main application with Streamlit UI
"""

import streamlit as st
import os
import sys
from typing import Optional
import logging
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# REMOVED: Direct imports of backend modules as logic is moved to API
# from data_processor import PhishingDataProcessor
# from insight_generator import InsightGenerator
# from embeddings import EmbeddingGenerator
# from vector_store import VectorStore, RAGRetriever
# from llm_orchestrator import LLMOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhishingChatbot:
    """
    Main chatbot application acting as a client to the API
    """
    
    def __init__(self):
        """Initialize the chatbot client"""
        self.api_url = os.getenv("API_URL", "http://localhost:8000")
        self.initialized = False
    
    def initialize(self, csv_path: str, groq_api_key: str):
        """
        Initialize connection to the API
        
        Args:
            csv_path: Path to CSV file (Unused in client mode, assumed API has data)
            groq_api_key: Groq API key (Unused in client mode, assumed API has key)
        """
        try:
            with st.spinner("Connecting to Agent API..."):
                # Check health of the API
                try:
                    response = requests.get(f"{self.api_url}/health")
                    response.raise_for_status()
                    data = response.json()

                    if data.get("initialized"):
                        self.initialized = True
                        st.success("✅ Connected to System API successfully!")
                        # Note: Data summary is skipped as we don't have direct access to DF
                        # self._show_data_summary()
                    else:
                        st.warning("⚠️ API is running but reports not initialized. Please check API logs.")

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Could not connect to API at {self.api_url}. Is it running?")
                except Exception as e:
                    st.error(f"❌ Error connecting to API: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error initializing chatbot client: {e}")
            st.error(f"Initialization failed: {str(e)}")
            # raise
    
    def _show_data_summary(self):
        """Display summary statistics"""
        # TODO: Implement API endpoint to fetch summary stats
        st.info("Data summary not available in API client mode.")
    
    def query(self, user_query: str) -> str:
        """
        Process user query via API
        
        Args:
            user_query: User's question
            
        Returns:
            Response string
        """
        if not self.initialized:
            return "System not connected to API. Please initialize first."
        
        try:
            payload = {"user_query": user_query}
            response = requests.post(f"{self.api_url}/query_agent", json=payload)

            if response.status_code == 200:
                return response.json().get("response", "No response field in JSON")
            else:
                return f"Error from API ({response.status_code}): {response.text}"
            
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
        
        # In API client mode, we still show this but it might be less relevant
        # if the API is configured via env vars on the server side.
        # But we keep it to maintain UI structure.

        if env_api_key:
            st.success("✅ API key loaded from .env file")
            groq_api_key = env_api_key
            # Show masked version
            masked_key = env_api_key[:8] + "..." + env_api_key[-4:] if len(env_api_key) > 12 else "***"
            st.caption(f"Using key: `{masked_key}`")
        else:
            # We don't strictly need it here if the API has it, but let's keep it consistent
            st.info("API Key should be configured on the backend.")
            groq_api_key = "configured-on-backend"
        
        # File upload
        st.header("📁 Data Upload")
        st.info("Data management is handled by the backend API in this version.")
        
        # Initialize button
        if st.button("🚀 Connect to System", type="primary"):
            # Initialize chatbot
            if 'chatbot' not in st.session_state:
                st.session_state.chatbot = PhishingChatbot()

            # We pass dummy values as the API handles the actual init
            st.session_state.chatbot.initialize("dummy_path.csv", groq_api_key)
        
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
            # Note: We can't easily clear backend history via this simple API yet
            st.rerun()
    
    else:
        # Welcome screen
        st.info("👈 Please connect to the system using the sidebar")
        
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

if __name__ == "__main__":
    main()
