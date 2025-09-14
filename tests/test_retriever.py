"""
Tests for RetrieverAgent pgvector functionality
"""

import pytest
from unittest.mock import patch, MagicMock
from agents.retriever_agent import RetrieverAgent
from schemas.review import RetrievalReport


class TestRetrieverAgent:
    """Test RetrieverAgent pgvector integration"""
    
    @pytest.fixture
    def retriever(self):
        return RetrieverAgent()
    
    @pytest.fixture
    def mock_db_results(self):
        """Mock database query results"""
        return [
            (1, 10, "Sample text content 1", 0.85),
            (2, 10, "Sample text content 2", 0.78), 
            (3, 11, "Sample text content 3", 0.65),
            (4, 11, "Sample text content 4", 0.60)
        ]
    
    @pytest.mark.asyncio
    async def test_process_returns_correct_structure(self, retriever, mock_db_results):
        """Test that process returns RetrievalReport with correct structure"""
        with patch('agents.retriever_agent.get_database_session') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.fetchall.return_value = mock_db_results
            
            with patch.object(retriever, '_get_embedding') as mock_embed:
                mock_embed.return_value = [0.1] * 768
                
                result = await retriever.process({"details": "test query"})
                
                assert isinstance(result, RetrievalReport)
                assert len(result.passages) == len(result.tags)
                assert len(result.passages) == 4
                assert len(result.doc_ids) == 2  # Unique doc IDs: 10, 11
                assert 0.0 <= result.coverage <= 1.0
    
    @pytest.mark.asyncio
    async def test_passages_trimmed_to_1500_chars(self, retriever):
        """Test that passages are trimmed to max 1500 chars"""
        long_text = "A" * 2000  # 2000 character string
        mock_results = [(1, 10, long_text, 0.85)]
        
        with patch('agents.retriever_agent.get_database_session') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.fetchall.return_value = mock_results
            
            with patch.object(retriever, '_get_embedding') as mock_embed:
                mock_embed.return_value = [0.1] * 768
                
                result = await retriever.process({"details": "test"})
                
                assert len(result.passages[0]) == 1500
    
    @pytest.mark.asyncio
    async def test_coverage_calculation(self, retriever, mock_db_results):
        """Test that coverage is calculated as mean similarity"""
        with patch('agents.retriever_agent.get_database_session') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.fetchall.return_value = mock_db_results
            
            with patch.object(retriever, '_get_embedding') as mock_embed:
                mock_embed.return_value = [0.1] * 768
                
                result = await retriever.process({"details": "test"})
                
                # Mean of [0.85, 0.78, 0.65, 0.60] = 0.72
                expected_coverage = (0.85 + 0.78 + 0.65 + 0.60) / 4
                assert result.coverage == round(expected_coverage, 3)
    
    @pytest.mark.asyncio
    async def test_tag_format_correct(self, retriever, mock_db_results):
        """Test that tags follow correct format: doc:<document_id>#chunk:<id>"""
        with patch('agents.retriever_agent.get_database_session') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.fetchall.return_value = mock_db_results
            
            with patch.object(retriever, '_get_embedding') as mock_embed:
                mock_embed.return_value = [0.1] * 768
                
                result = await retriever.process({"details": "test"})
                
                expected_tags = [
                    "doc:10#chunk:1",
                    "doc:10#chunk:2", 
                    "doc:11#chunk:3",
                    "doc:11#chunk:4"
                ]
                assert result.tags == expected_tags
    
    @pytest.mark.asyncio
    async def test_empty_results_returns_zero_coverage(self, retriever):
        """Test that empty results return zero coverage and empty arrays"""
        with patch('agents.retriever_agent.get_database_session') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.fetchall.return_value = []
            
            with patch.object(retriever, '_get_embedding') as mock_embed:
                mock_embed.return_value = [0.1] * 768
                
                result = await retriever.process({"details": "test"})
                
                assert result.passages == []
                assert result.tags == []
                assert result.doc_ids == []
                assert result.coverage == 0.0