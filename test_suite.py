"""
Test Suite for Phishing Campaign Analyzer
Run with: pytest test_suite.py -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json

from data_processor import PhishingDataProcessor
from insight_generator import InsightGenerator
from embeddings import EmbeddingGenerator
from vector_store import VectorStore, RAGRetriever


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def sample_csv():
    """Create a temporary CSV file for testing"""
    data = """User_ID,Department,Template,Action,Response_Time_Sec
U001,Finance,Urgent Password Reset,Clicked,35
U002,Sales,CEO Impersonation,Ignored,0
U003,IT,Fake Invoice,Reported,120
U004,Finance,Package Delivery,Clicked,45
U005,HR,Payroll Update,Clicked,28
U006,Marketing,LinkedIn Connection,Ignored,0
U007,Finance,Bank Alert,Clicked,15
U008,IT,Software Update,Reported,95
U009,Sales,Client Payment,Clicked,52
U010,HR,Benefits Enrollment,Ignored,0"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(data)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def data_processor(sample_csv):
    """Create a data processor instance"""
    processor = PhishingDataProcessor(sample_csv)
    processor.load_data()
    return processor


@pytest.fixture
def insight_generator(data_processor):
    """Create an insight generator instance"""
    return InsightGenerator(data_processor)


@pytest.fixture
def embedding_generator():
    """Create an embedding generator instance"""
    return EmbeddingGenerator()


# ============================================
# Data Processor Tests
# ============================================

class TestDataProcessor:
    """Test the PhishingDataProcessor class"""
    
    def test_load_data(self, sample_csv):
        """Test CSV data loading"""
        processor = PhishingDataProcessor(sample_csv)
        df = processor.load_data()
        
        assert df is not None
        assert len(df) == 10
        assert 'User_ID' in df.columns
        assert 'Department' in df.columns
    
    def test_calculate_click_rates(self, data_processor):
        """Test click rate calculation"""
        click_rates = data_processor.calculate_click_rates()
        
        assert isinstance(click_rates, dict)
        assert 'Finance' in click_rates
        assert 0 <= click_rates['Finance'] <= 1
    
    def test_calculate_template_effectiveness(self, data_processor):
        """Test template effectiveness analysis"""
        templates = data_processor.calculate_template_effectiveness()
        
        assert isinstance(templates, pd.DataFrame)
        assert 'click_rate' in templates.columns
        assert 'total_sent' in templates.columns
    
    def test_identify_high_risk_users(self, data_processor):
        """Test high-risk user identification"""
        high_risk = data_processor.identify_high_risk_users(top_n=3)
        
        assert isinstance(high_risk, pd.DataFrame)
        assert len(high_risk) <= 3
        assert 'risk_score' in high_risk.columns
    
    def test_analyze_response_times(self, data_processor):
        """Test response time analysis"""
        response_stats = data_processor.analyze_response_times()
        
        assert isinstance(response_stats, dict)
        assert 'overall_avg' in response_stats
        assert 'overall_median' in response_stats
        assert response_stats['overall_avg'] > 0
    
    def test_get_department_summary(self, data_processor):
        """Test department summary generation"""
        summary = data_processor.get_department_summary('Finance')
        
        assert isinstance(summary, dict)
        assert 'click_rate' in summary
        assert 'total_emails' in summary
        assert summary['total_emails'] > 0
    
    def test_query_interface(self, data_processor):
        """Test the query interface"""
        result = data_processor.query_data('click_rate')
        assert isinstance(result, dict)
        
        result = data_processor.query_data(
            'department_summary',
            department='Finance'
        )
        assert isinstance(result, dict)


# ============================================
# Insight Generator Tests
# ============================================

class TestInsightGenerator:
    """Test the InsightGenerator class"""
    
    def test_generate_department_insights(self, insight_generator):
        """Test department insight generation"""
        insights = insight_generator.generate_department_insights()
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        for insight in insights:
            assert 'insight_id' in insight
            assert 'category' in insight
            assert 'text' in insight
            assert insight['category'] == 'department_vulnerability'
    
    def test_generate_template_insights(self, insight_generator):
        """Test template insight generation"""
        insights = insight_generator.generate_template_insights()
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        for insight in insights:
            assert 'template' in insight
            assert 'effectiveness' in insight
            assert 'click_rate' in insight
    
    def test_generate_user_risk_insights(self, insight_generator):
        """Test user risk insight generation"""
        insights = insight_generator.generate_user_risk_insights(top_n=3)
        
        assert isinstance(insights, list)
        assert len(insights) <= 3
        
        for insight in insights:
            assert 'user_id' in insight
            assert 'risk_level' in insight
            assert 'risk_score' in insight
    
    def test_generate_behavioral_insights(self, insight_generator):
        """Test behavioral insight generation"""
        insights = insight_generator.generate_behavioral_insights()
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        for insight in insights:
            assert 'category' in insight
            assert insight['category'] == 'behavioral_insights'
    
    def test_generate_all_insights(self, insight_generator):
        """Test complete insight generation"""
        insights = insight_generator.generate_all_insights()
        
        assert isinstance(insights, list)
        assert len(insights) > 5  # Should have multiple types
        
        # Check for different categories
        categories = set(i['category'] for i in insights)
        assert 'department_vulnerability' in categories
        assert 'template_effectiveness' in categories
    
    def test_save_and_load_insights(self, insight_generator):
        """Test saving insights to file"""
        insights = insight_generator.generate_all_insights()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        insight_generator.save_insights(temp_path)
        
        assert os.path.exists(temp_path)
        
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        assert len(loaded) == len(insights)
        
        os.unlink(temp_path)


# ============================================
# Embedding Generator Tests
# ============================================

class TestEmbeddingGenerator:
    """Test the EmbeddingGenerator class"""
    
    def test_load_model(self, embedding_generator):
        """Test model loading"""
        embedding_generator.load_model()
        
        assert embedding_generator.model is not None
        assert embedding_generator.dimension > 0
    
    def test_encode_single_text(self, embedding_generator):
        """Test encoding a single text"""
        text = "Finance department shows high vulnerability"
        embedding = embedding_generator.encode_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == embedding_generator.get_dimension()
        assert -1 <= embedding[0] <= 1  # Normalized
    
    def test_encode_multiple_texts(self, embedding_generator):
        """Test encoding multiple texts"""
        texts = [
            "Finance is vulnerable",
            "Sales has low click rate",
            "IT reports phishing"
        ]
        embeddings = embedding_generator.encode_text(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 3
        assert embeddings.shape[1] == embedding_generator.get_dimension()
    
    def test_compute_similarity(self, embedding_generator):
        """Test similarity computation"""
        text1 = "Finance department vulnerability"
        text2 = "Finance shows high risk"
        text3 = "Sales team performance"
        
        sim_similar = embedding_generator.compute_similarity(text1, text2)
        sim_different = embedding_generator.compute_similarity(text1, text3)
        
        assert 0 <= sim_similar <= 1
        assert 0 <= sim_different <= 1
        assert sim_similar > sim_different  # Similar texts should have higher score
    
    def test_encode_insights(self, embedding_generator, insight_generator):
        """Test encoding insights with embeddings"""
        insights = insight_generator.generate_all_insights()[:5]
        insights_with_embeddings = embedding_generator.encode_insights(insights)
        
        assert len(insights_with_embeddings) == 5
        
        for insight in insights_with_embeddings:
            assert 'embedding' in insight
            assert isinstance(insight['embedding'], list)
            assert len(insight['embedding']) == embedding_generator.get_dimension()


# ============================================
# Vector Store Tests
# ============================================

class TestVectorStore:
    """Test the VectorStore class"""
    
    def test_connect(self):
        """Test connection to Qdrant"""
        vector_store = VectorStore()
        vector_store.connect()
        
        assert vector_store.client is not None
    
    def test_create_collection(self):
        """Test collection creation"""
        vector_store = VectorStore(collection_name="test_collection")
        vector_store.connect()
        vector_store.create_collection(recreate=True)
        
        # Should not raise an error
        info = vector_store.get_collection_info()
        assert 'name' in info
    
    def test_add_and_search_insights(self, embedding_generator, insight_generator):
        """Test adding and searching insights"""
        # Generate insights with embeddings
        insights = insight_generator.generate_all_insights()[:5]
        insights_with_embeddings = embedding_generator.encode_insights(insights)
        
        # Create vector store
        vector_store = VectorStore(collection_name="test_search")
        vector_store.connect()
        vector_store.create_collection(recreate=True)
        
        # Add insights
        vector_store.add_insights(insights_with_embeddings)
        
        # Search
        query = "Finance department vulnerability"
        query_embedding = embedding_generator.encode_text(query)
        results = vector_store.search(query_embedding.tolist(), top_k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        for result in results:
            assert 'score' in result
            assert 'insight' in result
            assert 0 <= result['score'] <= 1


class TestRAGRetriever:
    """Test the RAGRetriever class"""
    
    def test_retrieve_context(self, embedding_generator, insight_generator):
        """Test context retrieval"""
        # Setup
        insights = insight_generator.generate_all_insights()
        insights_with_embeddings = embedding_generator.encode_insights(insights)
        
        vector_store = VectorStore(collection_name="test_rag")
        vector_store.connect()
        vector_store.create_collection(recreate=True)
        vector_store.add_insights(insights_with_embeddings)
        
        rag_retriever = RAGRetriever(vector_store, embedding_generator)
        
        # Test retrieval
        query = "Which department is most vulnerable?"
        contexts = rag_retriever.retrieve_context(query, top_k=3)
        
        assert isinstance(contexts, list)
        assert len(contexts) <= 3
    
    def test_format_context_for_llm(self, embedding_generator, insight_generator):
        """Test context formatting"""
        # Setup
        insights = insight_generator.generate_all_insights()[:3]
        insights_with_embeddings = embedding_generator.encode_insights(insights)
        
        vector_store = VectorStore(collection_name="test_format")
        vector_store.connect()
        vector_store.create_collection(recreate=True)
        vector_store.add_insights(insights_with_embeddings)
        
        rag_retriever = RAGRetriever(vector_store, embedding_generator)
        
        # Get contexts
        query = "Security assessment"
        contexts = rag_retriever.retrieve_context(query, top_k=2)
        
        # Format
        formatted = rag_retriever.format_context_for_llm(contexts)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert 'Context' in formatted


# ============================================
# Integration Tests
# ============================================

class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_end_to_end_pipeline(self, sample_csv):
        """Test the complete pipeline from CSV to insights"""
        # 1. Load data
        processor = PhishingDataProcessor(sample_csv)
        processor.load_data()
        assert len(processor.df) > 0
        
        # 2. Generate insights
        generator = InsightGenerator(processor)
        insights = generator.generate_all_insights()
        assert len(insights) > 0
        
        # 3. Create embeddings
        embedding_gen = EmbeddingGenerator()
        insights_with_embeddings = embedding_gen.encode_insights(insights)
        assert all('embedding' in i for i in insights_with_embeddings)
        
        # 4. Store in vector DB
        vector_store = VectorStore(collection_name="test_e2e")
        vector_store.connect()
        vector_store.create_collection(recreate=True)
        vector_store.add_insights(insights_with_embeddings)
        
        # 5. Retrieve
        rag_retriever = RAGRetriever(vector_store, embedding_gen)
        contexts = rag_retriever.retrieve_context("What is the risk?", top_k=2)
        assert len(contexts) > 0


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
