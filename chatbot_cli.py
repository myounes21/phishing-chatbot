"""
Command-Line Interface for Phishing Campaign Chatbot
"""

import os
import sys
import argparse
from typing import Optional
import logging

from data_processor import PhishingDataProcessor
from insight_generator import InsightGenerator
from embeddings import EmbeddingGenerator
from vector_store import VectorStore, RAGRetriever
from llm_orchestrator import LLMOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLIChatbot:
    """Command-line interface for the chatbot"""
    
    def __init__(self, csv_path: str, groq_api_key: str):
        """
        Initialize CLI chatbot
        
        Args:
            csv_path: Path to CSV file
            groq_api_key: Groq API key
        """
        self.csv_path = csv_path
        self.groq_api_key = groq_api_key
        self.orchestrator = None
        
    def initialize(self):
        """Initialize all components"""
        print("\n" + "="*80)
        print("INITIALIZING PHISHING CAMPAIGN ANALYZER")
        print("="*80 + "\n")
        
        try:
            # Step 1: Load data
            print("📊 [1/6] Loading phishing campaign data...")
            data_processor = PhishingDataProcessor(self.csv_path)
            data_processor.load_data()
            print(f"    ✓ Loaded {len(data_processor.df)} records")
            
            # Step 2: Generate insights
            print("\n🧠 [2/6] Generating analytical insights...")
            insight_generator = InsightGenerator(data_processor)
            insights = insight_generator.generate_all_insights()
            print(f"    ✓ Generated {len(insights)} insights")
            
            # Step 3: Create embeddings
            print("\n🔢 [3/6] Creating text embeddings...")
            embedding_generator = EmbeddingGenerator()
            insights_with_embeddings = embedding_generator.encode_insights(insights)
            print(f"    ✓ Created embeddings (dimension: {embedding_generator.get_dimension()})")
            
            # Step 4: Setup vector store
            print("\n💾 [4/6] Setting up vector database...")
            vector_store = VectorStore()
            vector_store.connect()
            vector_store.create_collection(recreate=True)
            vector_store.add_insights(insights_with_embeddings)
            print(f"    ✓ Stored {len(insights)} vectors in Qdrant")
            
            # Step 5: Create RAG retriever
            print("\n🔍 [5/6] Configuring RAG system...")
            rag_retriever = RAGRetriever(vector_store, embedding_generator)
            print("    ✓ RAG retriever ready")
            
            # Step 6: Initialize orchestrator
            print("\n🤖 [6/6] Connecting to Groq LLM...")
            self.orchestrator = LLMOrchestrator(
                api_key=self.groq_api_key,
                data_processor=data_processor,
                rag_retriever=rag_retriever
            )
            print("    ✓ LLM orchestrator initialized")
            
            print("\n" + "="*80)
            print("✅ SYSTEM READY!")
            print("="*80 + "\n")
            
            # Show data summary
            self._show_summary(data_processor)
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            print(f"\n❌ Error: {str(e)}")
            return False
    
    def _show_summary(self, data_processor):
        """Display data summary"""
        df = data_processor.df
        
        total = len(df)
        users = df['User_ID'].nunique()
        depts = df['Department'].nunique()
        clicked = len(df[df['Action'] == 'Clicked'])
        reported = len(df[df['Action'] == 'Reported'])
        
        click_rate = (clicked / total) * 100
        report_rate = (reported / total) * 100
        
        print("📈 CAMPAIGN SUMMARY")
        print("-" * 80)
        print(f"Total Emails Sent:    {total}")
        print(f"Unique Users:         {users}")
        print(f"Departments:          {depts}")
        print(f"Clicked:              {clicked} ({click_rate:.1f}%)")
        print(f"Reported:             {reported} ({report_rate:.1f}%)")
        print("-" * 80 + "\n")
    
    def run_interactive(self):
        """Run interactive chat loop"""
        print("💬 INTERACTIVE MODE")
        print("Type your questions (or 'quit', 'exit', 'help' for commands)")
        print("-" * 80 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'clear':
                    self.orchestrator.clear_history()
                    print("🗑️  Chat history cleared")
                    continue
                
                if user_input.lower() == 'examples':
                    self._show_examples()
                    continue
                
                # Process query
                print("\n🤖 Analyzing", end="", flush=True)
                for _ in range(3):
                    print(".", end="", flush=True)
                    
                response = self.orchestrator.process_query(user_input)
                
                print("\n")
                print("💡 Assistant:")
                print("-" * 80)
                print(response)
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\n❌ Error: {str(e)}")
    
    def _show_help(self):
        """Show help message"""
        print("\n" + "="*80)
        print("AVAILABLE COMMANDS")
        print("="*80)
        print("help      - Show this help message")
        print("examples  - Show example queries")
        print("clear     - Clear chat history")
        print("quit/exit - Exit the program")
        print("="*80)
    
    def _show_examples(self):
        """Show example queries"""
        print("\n" + "="*80)
        print("EXAMPLE QUERIES")
        print("="*80)
        print("\n📊 Quantitative Queries:")
        print("  - What is the click rate for Finance?")
        print("  - Who are the top 5 riskiest users?")
        print("  - Which templates are most effective?")
        print("  - What's the average response time?")
        print("\n🧠 Qualitative Queries:")
        print("  - Why is Finance vulnerable to phishing?")
        print("  - What psychological triggers does the urgent template use?")
        print("  - How can we improve security awareness?")
        print("  - What patterns do high-risk users show?")
        print("\n🔄 Synthesis Queries:")
        print("  - Give me a complete risk assessment")
        print("  - What are the biggest security gaps?")
        print("  - How do departments compare in security awareness?")
        print("="*80)
    
    def run_single_query(self, query: str):
        """
        Run a single query and exit
        
        Args:
            query: Query string
        """
        print(f"\n🤔 Query: {query}")
        print("-" * 80)
        
        response = self.orchestrator.process_query(query)
        
        print("\n💡 Response:")
        print("-" * 80)
        print(response)
        print("-" * 80 + "\n")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Phishing Campaign Analyzer - Intelligent Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python chatbot_cli.py --csv data.csv --api-key YOUR_KEY
  
  # Single query
  python chatbot_cli.py --csv data.csv --api-key YOUR_KEY --query "What is the click rate?"
  
  # Use environment variable for API key
  export GROQ_API_KEY=your_key_here
  python chatbot_cli.py --csv data.csv
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with phishing campaign data'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Groq API key (or set GROQ_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Single query to execute (skips interactive mode)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='llama-3.3-70b-versatile',
        help='Groq model to use (default: llama-3.3-70b-versatile)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("❌ Error: Groq API key required!")
        print("   Provide via --api-key or set GROQ_API_KEY environment variable")
        sys.exit(1)
    
    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"❌ Error: CSV file not found: {args.csv}")
        sys.exit(1)
    
    # Create chatbot
    chatbot = CLIChatbot(args.csv, api_key)
    
    # Initialize
    if not chatbot.initialize():
        print("❌ Failed to initialize chatbot")
        sys.exit(1)
    
    # Run mode
    if args.query:
        # Single query mode
        chatbot.run_single_query(args.query)
    else:
        # Interactive mode
        chatbot.run_interactive()


if __name__ == "__main__":
    main()