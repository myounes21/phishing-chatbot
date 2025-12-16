"""
LLM Orchestrator Module
Handles query understanding and tool orchestration using Groq LLM
"""

import os
import json
from typing import Dict, List, Any, Optional
from groq import Groq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Orchestrates query processing using Groq LLM and appropriate tools
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "llama-3.3-70b-versatile",
                 data_processor=None,
                 rag_retriever=None):
        """
        Initialize the LLM orchestrator
        
        Args:
            api_key: Groq API key (or use GROQ_API_KEY env variable)
            model: Groq model to use (default: llama-3.3-70b-versatile)
                   Other options: llama-3.1-70b-versatile, llama-3.1-8b-instant
            data_processor: PhishingDataProcessor instance
            rag_retriever: RAGRetriever instance
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable.")
        
        self.model = model
        self.client = Groq(api_key=self.api_key)
        self.data_processor = data_processor
        self.rag_retriever = rag_retriever
        self.conversation_history = []
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM"""
        return """You are a versatile AI assistant specializing in Phishing Campaign Analysis and Organizational Knowledge.

You have access to two types of data sources:

1. **Phishing Campaign Data (Quantitative & Analytical)**
   - Accessed via **Pandas Tool** for exact numbers, stats, lists, and rankings.
   - Accessed via **RAG Tool** (category: `phishing_insight`) for qualitative insights, explanations, and behavioral patterns.
   - *Use when asking about:* Click rates, risky users, department performance, template stats, simulation results.

2. **Knowledge Base (Textual & Informational)**
   - Accessed via **RAG Tool** only.
   - Categories:
     - `org_knowledge`: Company policies, services, "who we are", internal procedures.
     - `general_knowledge`: General cybersecurity concepts, phishing definitions, best practices, educational content.
   - *Use when asking about:* Company info, security policies, definitions of terms, general advice.

**Instructions:**
- **Accuracy First:** Base your answers ONLY on the provided data/context. Do not hallucinate.
- **Data Selection:**
  - For specific numbers/stats about the *current campaign*, prioritize the **Pandas Tool**.
  - For explanations of those numbers, use **RAG (phishing_insight)**.
  - For general questions or company info, use **RAG (org_knowledge / general_knowledge)**.
- **Response Style:** Professional, clear, and actionable. Use bullet points for lists.

Available query types for Pandas tool:
- click_rate, department_click_rate, template_effectiveness, high_risk_users, response_times, department_summary, full_report
"""

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify the query to determine which tools to use
        
        Args:
            query: User query
            
        Returns:
            Dictionary with classification results
        """
        classification_prompt = f"""Classify this query and determine which tools are needed:

Query: "{query}"

Respond in JSON format:
{{
    "needs_pandas": true/false,
    "needs_rag": true/false,
    "query_type": "quantitative/qualitative/synthesis",
    "pandas_query_type": "click_rate/template_effectiveness/etc or null",
    "parameters": {{}} // Any specific parameters needed
}}

Examples:
- "What is the click rate for Finance?" -> {{"needs_pandas": true, "needs_rag": false, "query_type": "quantitative"}}
- "Why did the urgent template work well?" -> {{"needs_pandas": false, "needs_rag": true, "query_type": "qualitative"}}
- "Who are the riskiest users and why?" -> {{"needs_pandas": true, "needs_rag": true, "query_type": "synthesis"}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query classifier. Respond only with valid JSON."},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Query classified as: {result['query_type']}")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default fallback
            return {
                "needs_pandas": True,
                "needs_rag": False,
                "query_type": "quantitative"
            }
    
    def _execute_pandas_query(self, query_type: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a Pandas query
        
        Args:
            query_type: Type of query to execute
            parameters: Query parameters
            
        Returns:
            Query results
        """
        if not self.data_processor:
            return {"error": "Data processor not available"}
        
        try:
            result = self.data_processor.query_data(query_type, **parameters)
            logger.info(f"Executed Pandas query: {query_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing Pandas query: {e}")
            return {"error": str(e)}
    
    def _execute_rag_query(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute a RAG query
        
        Args:
            query: User query
            category: Optional category filter
            
        Returns:
            Retrieved contexts
        """
        if not self.rag_retriever:
            return [{"error": "RAG retriever not available"}]
        
        try:
            contexts = self.rag_retriever.retrieve_context(query, top_k=5, category=category)
            logger.info(f"Retrieved {len(contexts)} contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error executing RAG query: {e}")
            return [{"error": str(e)}]
    
    def _format_data_for_llm(self, data: Any) -> str:
        """
        Format data for LLM consumption
        
        Args:
            data: Data to format (dict, dataframe, list, etc.)
            
        Returns:
            Formatted string
        """
        import pandas as pd
        
        if isinstance(data, pd.DataFrame):
            return data.to_string()
        elif isinstance(data, dict):
            return json.dumps(data, indent=2)
        elif isinstance(data, list):
            return json.dumps(data, indent=2)
        else:
            return str(data)
    
    def process_query(self, query: str) -> str:
        """
        Process user query and generate response
        
        Args:
            query: User query
            
        Returns:
            Response string
        """
        try:
            # Classify query
            classification = self._classify_query(query)
            
            # Collect data from tools
            context_parts = []
            
            if classification.get('needs_pandas', False):
                pandas_query_type = classification.get('pandas_query_type', 'full_report')
                parameters = classification.get('parameters', {})
                
                pandas_result = self._execute_pandas_query(pandas_query_type, parameters)
                
                context_parts.append(
                    f"=== QUANTITATIVE DATA ===\n"
                    f"{self._format_data_for_llm(pandas_result)}\n"
                )
            
            if classification.get('needs_rag', False):
                # We can use the classified category to narrow down search if needed
                # For now, we might want to search broadly or specific based on implementation
                # The current RAG retriever supports 'category' which matches our 'source_type' or 'category' field?
                # Let's assume the classifier's 'rag_category' maps to the 'source_type' or 'category' in Qdrant

                rag_category = classification.get('rag_category')
                # If rag_category is one of our known types, we pass it.
                # Note: vector_store.py has `search_by_source_type` and `search_by_category`.
                # We should probably align these.

                # In vector_store.py we have `search_by_category` filtering on 'category' field.
                # In document_processor.py we set 'category' = doc_type ('org_knowledge', etc).
                # In insight_generator.py we set 'category' = 'department_vulnerability' etc, and 'source_type' = 'phishing_insight'.

                # So:
                # If asking about phishing insights, we might want to filter by source_type='phishing_insight'
                # If asking about org knowledge, filter by category='org_knowledge'

                # To simplify, let's just pass the query to RAG. The RAG retriever in `api_backend` calls `search_by_category` if category is passed.
                # But our categories are mixed (some are 'department_vulnerability', some are 'org_knowledge').

                # Let's just search broadly for now, OR:
                # If the classifier is confident about 'org_knowledge' or 'general_knowledge', we pass that as category.
                # If it's 'phishing_insight', we might want to search by source_type, but `retrieve_context` only takes `category`.

                # Let's adjust based on what we have.
                # The current `retrieve_context` uses `search_by_category`.
                # If I pass 'org_knowledge', it will work for those docs.
                # If I pass 'phishing_insight', it won't work because phishing insights have granular categories.

                # Better approach: Just search without filter, let similarity decide?
                # Or update `retrieve_context` to handle this.
                # For this iteration, I will just search without category to let semantic similarity work,
                # unless I want to be strict.

                # Let's try passing it if it is org/general, but ignore if phishing_insight

                category_filter = None
                if rag_category in ['org_knowledge', 'general_knowledge']:
                    category_filter = rag_category

                rag_results = self._execute_rag_query(query, category=category_filter)
                
                if rag_results and 'error' not in rag_results[0]:
                    formatted_context = self.rag_retriever.format_context_for_llm(rag_results)
                    context_parts.append(
                        f"=== QUALITATIVE INSIGHTS / KNOWLEDGE ===\n"
                        f"{formatted_context}\n"
                    )
            
            # Combine contexts
            full_context = "\n".join(context_parts) if context_parts else "No specific data available."
            
            # Generate response using LLM
            response = self._generate_response(query, full_context)
            
            # Add to conversation history
            self.conversation_history.append({
                "query": query,
                "response": response
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error processing your query: {str(e)}"
    
    def _generate_response(self, query: str, context: str) -> str:
        """
        Generate final response using LLM
        
        Args:
            query: User query
            context: Collected context from tools
            
        Returns:
            Generated response
        """
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
        ]
        
        # Add recent conversation history (last 3 exchanges)
        for exchange in self.conversation_history[-3:]:
            messages.append({"role": "user", "content": exchange["query"]})
            messages.append({"role": "assistant", "content": exchange["response"]})
        
        # Add current query with context
        user_message = f"""Query: {query}

Available Data and Context:
{context}

Please provide a clear, actionable response based on this data. Use specific numbers and insights from the context."""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")


if __name__ == "__main__":
    # Test the orchestrator (requires API key)
    from data_processor import PhishingDataProcessor
    from insight_generator import InsightGenerator
    from embeddings import EmbeddingGenerator
    from vector_store import VectorStore, RAGRetriever
    
    print("Note: This test requires GROQ_API_KEY environment variable to be set.")
    
    if not os.getenv('GROQ_API_KEY'):
        print("GROQ_API_KEY not found. Please set it to test the orchestrator.")
        print("\nExample usage:")
        print("  export GROQ_API_KEY='your-api-key-here'")
        print("  python llm_orchestrator.py")
    else:
        # Setup components
        processor = PhishingDataProcessor("sample_phishing_data.csv")
        processor.load_data()
        
        generator = InsightGenerator(processor)
        insights = generator.generate_all_insights()
        
        embedding_gen = EmbeddingGenerator()
        insights_with_embeddings = embedding_gen.encode_insights(insights)
        
        vector_store = VectorStore()
        vector_store.connect()
        vector_store.create_collection(recreate=True)
        vector_store.add_insights(insights_with_embeddings)
        
        rag_retriever = RAGRetriever(vector_store, embedding_gen)
        
        # Create orchestrator
        orchestrator = LLMOrchestrator(
            data_processor=processor,
            rag_retriever=rag_retriever
        )
        
        # Test queries
        test_queries = [
            "What is the click rate for Finance department?",
            "Why is Finance vulnerable to phishing?",
            "Who are the top 5 riskiest users and why?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")
            
            response = orchestrator.process_query(query)
            print(response)