"""
Enhanced LLM Orchestrator Module
Generates detailed, well-structured responses using Groq LLM and RAG
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from groq import AsyncGroq
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
        # Higher temperature for more natural, conversational responses
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '2048'))
        
        self.client = AsyncGroq(api_key=self.api_key)
        self.rag_retriever = rag_retriever
        self.data_processor = data_processor
        self.conversation_history = []
        
    def _create_system_prompt(self) -> str:
        """Create natural, conversational system prompt"""
        return """You are a friendly and helpful AI assistant specializing in cybersecurity and phishing awareness. You help people understand phishing attacks, analyze security data, and learn about their organization.

**Your Personality:**
- Natural and conversational, like chatting with a knowledgeable friend
- Friendly, approachable, and genuinely helpful
- Enthusiastic about security education but never condescending
- Use everyday language, not overly technical jargon
- Show empathy when discussing security concerns

**Response Style:**
- Answer naturally and conversationally - like you're explaining to a friend
- Match the length to the question: simple questions get concise answers, complex ones get more detail
- Use "you" and "we" to make it personal and engaging
- Feel free to use casual phrases like "Here's the thing..." or "So basically..."
- Break up long paragraphs - keep it readable
- Use bullet points or lists when helpful, but don't overdo formatting
- Be concise when possible, but thorough when needed

**What You Know About:**
- Phishing campaigns: analyzing click rates, identifying risky users/departments, understanding attack patterns
- Company information: mission, values, services, team details
- Cybersecurity: phishing tactics, defense strategies, security best practices

**Guidelines:**
- Always base your answers on the provided context/data when available
- If you see specific numbers or data, mention them naturally (e.g., "I see the Finance department has a 25% click rate...")
- Explain things clearly but don't over-explain simple concepts
- Be helpful and offer practical advice when relevant
- If you don't know something, say so honestly

**Tone Examples:**
✅ Good: "Phishing is basically when scammers try to trick you into giving away sensitive info..."
✅ Good: "Looking at your data, I notice the Finance team has a higher click rate..."
✅ Good: "Here's what I found about your company..."

❌ Avoid: "Phishing is a type of cyber attack wherein malicious actors..."
❌ Avoid: "According to the data analysis, it can be observed that..."
❌ Avoid: Overly formal or robotic language

Remember: You're here to help, not to impress. Keep it natural, friendly, and genuinely useful!"""
    
    async def process_query(self, 
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
            
            # Retrieve relevant context (auto-detect is handled by RAGRetriever)
            logger.info("🔍 Retrieving context from vector database...")
            rag_top_k = int(os.getenv('RAG_TOP_K', '8'))
            contexts = await self.rag_retriever.retrieve_context(
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
            response_text = await self._generate_detailed_response(query, formatted_context)
            
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
    
    async def _generate_detailed_response(self, query: str, context: str) -> str:
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
        
        # Add current query with context - keep it natural
        if context and context.strip() != "No relevant context found.":
            user_message = f"""User asked: "{query}"

Here's some relevant information I found:
{context}

Please answer the user's question naturally and conversationally. Use the context above to inform your answer, but don't feel like you need to mention every detail. Just give a helpful, friendly response that directly addresses what they're asking. Keep it natural - like you're explaining to a friend."""
        else:
            user_message = f"""User asked: "{query}"

Answer this question naturally and conversationally. Be helpful and friendly, like you're chatting with someone who needs information."""

        messages.append({"role": "user", "content": user_message})
        
        try:
            # Call Groq API with configurable parameters from environment
            # Higher top_p for more natural, diverse responses
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.95,  # Higher for more natural language
                frequency_penalty=0.1,  # Slight penalty to avoid repetition
                presence_penalty=0.1  # Encourage more diverse topics
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
    
    async def generate_summary(self, campaign_id: Optional[str] = None) -> str:
        """
        Generate a comprehensive campaign summary
        
        Args:
            campaign_id: Specific campaign to summarize
            
        Returns:
            Detailed summary text
        """
        query = f"Provide a comprehensive summary and analysis of campaign {campaign_id}" if campaign_id else "Provide a comprehensive summary of all phishing campaigns"
        
        response, _ = await self.process_query(query, collection="phishing_insights", include_sources=False)
        return response
    
    async def explain_concept(self, concept: str) -> str:
        """
        Explain a phishing or security concept in detail
        
        Args:
            concept: Concept to explain
            
        Returns:
            Detailed explanation
        """
        query = f"Explain in detail: {concept}"
        response, _ = await self.process_query(query, collection="phishing_general", include_sources=False)
        return response
    
    async def company_info(self, question: str) -> str:
        """
        Answer questions about the company
        
        Args:
            question: Question about the company
            
        Returns:
            Detailed answer
        """
        response, _ = await self.process_query(question, collection="company_knowledge", include_sources=False)
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("🗑️ Conversation history cleared")


if __name__ == "__main__":
    import asyncio
    
    # Test the orchestrator
    print("Note: This test requires GROQ_API_KEY environment variable")
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ GROQ_API_KEY not found")
        exit(1)
        
    async def main():
        # This is a minimal test - full test requires RAG retriever
        orchestrator = LLMOrchestrator(api_key=api_key)
        print("✅ LLM Orchestrator initialized")
        
        # We can't really test query processing without a RAG retriever or mocking it
        # But we can check if the object is created correctly
        
    asyncio.run(main())
