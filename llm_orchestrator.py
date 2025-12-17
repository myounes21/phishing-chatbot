"""
Enhanced LLM Orchestrator Module
Generates detailed, well-structured responses using Groq LLM and RAG
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from groq import Groq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Enhanced orchestrator that generates detailed, human-friendly responses
    Supports multi-domain queries across phishing, company, and general knowledge
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 rag_retriever=None,
                 data_processor=None):
        """
        Initialize the LLM orchestrator
        
        Args:
            api_key: Groq API key
            model: Groq model to use (defaults to GROQ_MODEL env var or "llama-3.3-70b-versatile")
            rag_retriever: RAGRetriever instance
            data_processor: Optional PhishingDataProcessor for quantitative queries
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key required")
        
        self.model = model or os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.4'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '4096'))
        
        self.client = Groq(api_key=self.api_key)
        self.rag_retriever = rag_retriever
        self.data_processor = data_processor
        self.conversation_history = []
        
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt for detailed responses"""
        return """You are an expert AI assistant specializing in:
1. **Phishing Campaign Analysis** - Analyzing security simulations and user behavior
2. **Organizational Knowledge** - Answering questions about the company and its services
3. **Cybersecurity Education** - Explaining phishing tactics, defenses, and best practices

**Response Guidelines:**

1. **Comprehensive & Detailed**
   - Provide LONG, well-structured responses (300-800 words when appropriate)
   - Break down complex topics into clear sections
   - Include specific examples and actionable insights
   - Use analogies and explanations for clarity

2. **Professional Formatting**
   - Use markdown formatting: headers, bullet points, numbered lists
   - Organize information logically with clear sections
   - Highlight key points and statistics
   - Include relevant context and background

3. **Data-Driven Analysis**
   - Base all phishing analysis on provided data/context
   - Cite specific numbers, percentages, and metrics
   - Explain WHY patterns exist, not just WHAT they are
   - Connect insights to actionable recommendations

4. **Educational Approach**
   - Explain concepts thoroughly for non-technical audiences
   - Provide background information when relevant
   - Offer practical advice and next steps
   - Anticipate follow-up questions

5. **Tone & Style**
   - Professional but approachable
   - Clear and easy to understand
   - Empathetic when discussing vulnerabilities
   - Solution-focused and constructive

**Knowledge Domains:**

- **Phishing Campaigns**: Click rates, vulnerable users/departments, template effectiveness, behavioral patterns, risk scoring
- **Company Info**: Mission, values, services, team, history, achievements
- **Phishing General**: Tactics (urgency, authority, fear), defense strategies, awareness training, incident response

**Important**: Always structure responses with:
- Opening summary
- Detailed analysis with subsections
- Key takeaways or recommendations
- Closing with next steps or further assistance offer
"""

    def _detect_query_type(self, query: str) -> Dict[str, Any]:
        """
        Detect query type and determine which collections to search
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query metadata
        """
        query_lower = query.lower()
        
        # Detection keywords
        phishing_keywords = ['click rate', 'department', 'vulnerable', 'risk', 'template', 
                            'user', 'campaign', 'phishing simulation', 'security awareness']
        company_keywords = ['who are we', 'what do', 'company', 'organization', 'mission',
                           'values', 'services', 'about us', 'team']
        general_keywords = ['what is phishing', 'how to', 'defense', 'tactics', 'prevent',
                           'protect', 'awareness', 'training', 'best practices']
        pdf_keywords = ['document', 'pdf', 'file', 'report', 'policy', 'procedure', 
                       'manual', 'guide', 'specification', 'contract']
        
        # Count keyword matches
        phishing_score = sum(1 for kw in phishing_keywords if kw in query_lower)
        company_score = sum(1 for kw in company_keywords if kw in query_lower)
        general_score = sum(1 for kw in general_keywords if kw in query_lower)
        pdf_score = sum(1 for kw in pdf_keywords if kw in query_lower)
        
        # Determine primary type
        scores = {
            'phishing_insights': phishing_score,
            'company_knowledge': company_score,
            'phishing_general': general_score,
            'pdf_documents': pdf_score
        }
        
        primary_type = max(scores, key=scores.get)
        
        # If scores are similar, search all
        search_all = max(scores.values()) - min(scores.values()) <= 1
        
        return {
            'primary_collection': primary_type if not search_all else None,
            'search_all_collections': search_all,
            'confidence': max(scores.values()),
            'query_complexity': 'complex' if len(query.split()) > 15 else 'simple'
        }
    
    def process_query(self, 
                     query: str,
                     collection: Optional[str] = None,
                     include_sources: bool = True) -> Tuple[str, Optional[List[Dict]]]:
        """
        Process user query and generate detailed response
        
        Args:
            query: User query
            collection: Specific collection to search (or None for auto-detect)
            include_sources: Whether to return source documents
            
        Returns:
            Tuple of (response_text, sources)
        """
        try:
            logger.info(f"🤔 Processing query: {query[:100]}...")
            
            # Detect query type if collection not specified
            if not collection:
                query_metadata = self._detect_query_type(query)
                collection = query_metadata['primary_collection']
                logger.info(f"📊 Detected primary collection: {collection}")
            
            # Retrieve relevant context
            logger.info("🔍 Retrieving context from vector database...")
            rag_top_k = int(os.getenv('RAG_TOP_K', '8'))
            contexts = self.rag_retriever.retrieve_context(
                query=query,
                collection=collection,
                top_k=rag_top_k  # Get more contexts for comprehensive responses
            )
            
            if not contexts:
                logger.warning("⚠️ No contexts found, using general knowledge")
                contexts = []
            
            # Format context for LLM
            formatted_context = self.rag_retriever.format_context_for_llm(contexts)
            
            # Generate response
            logger.info("🤖 Generating detailed response...")
            response_text = self._generate_detailed_response(query, formatted_context)
            
            # Prepare sources for return
            sources = None
            if include_sources and contexts:
                sources = [
                    {
                        'content': ctx['payload'].get('text', '')[:200] + '...',
                        'relevance': ctx['score'],
                        'collection': ctx.get('collection', 'unknown'),
                        'title': ctx['payload'].get('title', ctx['payload'].get('campaign_name', 'Untitled'))
                    }
                    for ctx in contexts[:5]  # Return top 5 sources
                ]
            
            logger.info("✅ Response generated successfully")
            return response_text, sources
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return f"I encountered an error processing your query: {str(e)}", None
    
    def _generate_detailed_response(self, query: str, context: str) -> str:
        """
        Generate a detailed, well-structured response
        
        Args:
            query: User query
            context: Formatted context from RAG
            
        Returns:
            Generated response text
        """
        # Build messages
        messages = [
            {"role": "system", "content": self._create_system_prompt()}
        ]
        
        # Add recent conversation history (last 2 exchanges)
        for exchange in self.conversation_history[-2:]:
            messages.append({"role": "user", "content": exchange["query"]})
            messages.append({"role": "assistant", "content": exchange["response"]})
        
        # Add current query with context
        user_message = f"""User Question: {query}

Retrieved Context and Data:
{context}

Please provide a comprehensive, detailed response that:
1. Directly answers the question
2. Explains the reasoning and insights behind the data
3. Provides actionable recommendations when appropriate
4. Uses clear structure with headers and bullet points
5. Is thorough and educational (aim for 300-800 words)

Make sure to cite specific data points from the context when relevant."""

        messages.append({"role": "user", "content": user_message})
        
        try:
            # Call Groq API with configurable parameters from environment
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.9
            )
            
            response_text = response.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({
                "query": query,
                "response": response_text
            })
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response_text
            
        except Exception as e:
            logger.error(f"❌ Error calling Groq API: {e}")
            raise
    
    def generate_summary(self, campaign_id: Optional[str] = None) -> str:
        """
        Generate a comprehensive campaign summary
        
        Args:
            campaign_id: Specific campaign to summarize
            
        Returns:
            Detailed summary text
        """
        query = f"Provide a comprehensive summary and analysis of campaign {campaign_id}" if campaign_id else "Provide a comprehensive summary of all phishing campaigns"
        
        response, _ = self.process_query(query, collection="phishing_insights", include_sources=False)
        return response
    
    def explain_concept(self, concept: str) -> str:
        """
        Explain a phishing or security concept in detail
        
        Args:
            concept: Concept to explain
            
        Returns:
            Detailed explanation
        """
        query = f"Explain in detail: {concept}"
        response, _ = self.process_query(query, collection="phishing_general", include_sources=False)
        return response
    
    def company_info(self, question: str) -> str:
        """
        Answer questions about the company
        
        Args:
            question: Question about the company
            
        Returns:
            Detailed answer
        """
        response, _ = self.process_query(question, collection="company_knowledge", include_sources=False)
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("🗑️ Conversation history cleared")


if __name__ == "__main__":
    # Test the orchestrator
    print("Note: This test requires GROQ_API_KEY environment variable")
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ GROQ_API_KEY not found")
        exit(1)
    
    # This is a minimal test - full test requires RAG retriever
    orchestrator = LLMOrchestrator(api_key=api_key)
    print("✅ LLM Orchestrator initialized")
    
    # Test query detection
    test_queries = [
        "What is the click rate for Finance department?",
        "Who are we as a company?",
        "How can employees defend against phishing?"
    ]
    
    for query in test_queries:
        metadata = orchestrator._detect_query_type(query)
        print(f"\nQuery: {query}")
        print(f"Primary Collection: {metadata['primary_collection']}")