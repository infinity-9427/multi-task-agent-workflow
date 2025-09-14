"""
Tests for orchestration flow and coverage gates
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from main import app
from rag.orchestrator import RAGOrchestrator
from schemas.review import RetrievalReport


class TestOrchestration:
    """Test orchestration order and coverage gates"""
    
    @pytest.fixture
    def orchestrator(self):
        return RAGOrchestrator()
    
    @pytest.fixture  
    def mock_retrieval_low_coverage(self):
        """Mock retrieval with low coverage"""
        return RetrievalReport(
            passages=["Some text"],
            tags=["doc:1#chunk:1"],
            doc_ids=[1],
            coverage=0.2  # Below threshold
        )
    
    @pytest.fixture
    def mock_retrieval_high_coverage(self):
        """Mock retrieval with high coverage"""
        return RetrievalReport(
            passages=["Detailed policy text", "More context"],
            tags=["doc:1#chunk:1", "doc:2#chunk:5"],
            doc_ids=[1, 2],
            coverage=0.7  # Above threshold
        )
    
    @pytest.mark.asyncio
    async def test_coverage_gate_blocks_low_coverage(self, orchestrator, mock_retrieval_low_coverage):
        """Test that coverage gate blocks calls to DecisionAgent when coverage < threshold"""
        with patch.object(orchestrator.retriever, 'process') as mock_retriever:
            mock_retriever.return_value = mock_retrieval_low_coverage
            
            with patch.object(orchestrator.decision_agent, 'process') as mock_decision:
                # Should not be called due to coverage gate
                mock_decision.return_value = {"decision": "approve"}
                
                result = await orchestrator.process_review("test_id", "test details")
                
                # Verify DecisionAgent was NOT called
                mock_decision.assert_not_called()
                
                # Verify response is rejection due to low coverage
                assert result["data"]["decision"] == "reject"
                assert result["data"]["coverage"] == 0.2
                assert "below required threshold" in result["data"]["rationale"]
    
    @pytest.mark.asyncio  
    async def test_coverage_gate_allows_high_coverage(self, orchestrator, mock_retrieval_high_coverage):
        """Test that coverage gate allows calls to DecisionAgent when coverage >= threshold"""
        with patch.object(orchestrator.retriever, 'process') as mock_retriever:
            mock_retriever.return_value = mock_retrieval_high_coverage
            
            with patch.object(orchestrator.decision_agent, 'process') as mock_decision:
                mock_decision.return_value = {
                    "decision": "approve",
                    "rationale": "Test rationale",
                    "citations": ["doc:1#chunk:1", "doc:2#chunk:5"],
                    "confidence": 0.8,
                    "required_actions": []
                }
                
                result = await orchestrator.process_review("test_id", "test details")
                
                # Verify DecisionAgent WAS called
                mock_decision.assert_called_once()
                
                # Verify correct input passed to DecisionAgent
                call_args = mock_decision.call_args[0][0]
                assert call_args["details"] == "test details"
                assert len(call_args["passages"]) == 2
                assert len(call_args["tags"]) == 2
                assert call_args["coverage"] == 0.7
    
    @pytest.mark.asyncio
    async def test_policy_gate_requires_sufficient_citations(self, orchestrator, mock_retrieval_high_coverage):
        """Test policy gate requires >= 2 distinct citations for approval"""
        with patch.object(orchestrator.retriever, 'process') as mock_retriever:
            mock_retriever.return_value = mock_retrieval_high_coverage
            
            with patch.object(orchestrator.decision_agent, 'process') as mock_decision:
                # Decision with only 1 citation
                mock_decision.return_value = {
                    "decision": "approve", 
                    "rationale": "Test rationale",
                    "citations": ["doc:1#chunk:1"],  # Only 1 citation
                    "confidence": 0.8,
                    "required_actions": []
                }
                
                result = await orchestrator.process_review("test_id", "test details")
                
                # Should be downgraded to reject due to policy gate
                assert result["data"]["decision"] == "reject"
                assert "Insufficient context or citations" in result["data"]["rationale"]
    
    @pytest.mark.asyncio
    async def test_policy_gate_allows_sufficient_citations(self, orchestrator, mock_retrieval_high_coverage):
        """Test policy gate allows approval with >= 2 distinct citations and good coverage"""
        with patch.object(orchestrator.retriever, 'process') as mock_retriever:
            mock_retriever.return_value = mock_retrieval_high_coverage
            
            with patch.object(orchestrator.decision_agent, 'process') as mock_decision:
                # Decision with 2+ citations and good coverage
                mock_decision.return_value = {
                    "decision": "approve",
                    "rationale": "Test rationale", 
                    "citations": ["doc:1#chunk:1", "doc:2#chunk:5"],  # 2 distinct citations
                    "confidence": 0.8,
                    "required_actions": []
                }
                
                result = await orchestrator.process_review("test_id", "test details")
                
                # Should pass policy gate
                assert result["data"]["decision"] == "approve"
                assert len(result["data"]["citations"]) == 2


class TestAPIIntegration:
    """Test API integration with TestClient"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_review_endpoint_exists(self, client):
        """Test that review endpoint is accessible"""
        response = client.post("/review", json={
            "task_id": "test-001",
            "details": "simple test task"
        })
        
        # Should not return 404
        assert response.status_code != 404
    
    def test_health_endpoint_works(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        
        # Health endpoint should work (may fail on DB but shouldn't 404)
        assert response.status_code in [200, 503]  # OK or Service Unavailable
    
    def test_review_request_validation(self, client):
        """Test request validation"""
        # Missing task_id
        response = client.post("/review", json={"details": "test"})
        assert response.status_code == 422
        
        # Missing details
        response = client.post("/review", json={"task_id": "test-001"})
        assert response.status_code == 422
        
        # Empty details
        response = client.post("/review", json={"task_id": "test-001", "details": ""})
        assert response.status_code == 422
    
    def test_review_response_format(self, client):
        """Test that response follows envelope format (if DB available)"""
        response = client.post("/review", json={
            "task_id": "format-test-001",
            "details": "test task for response format validation"
        })
        
        if response.status_code == 200:
            data = response.json()
            
            # Should have envelope format
            assert "message" in data
            assert "data" in data
            
            # Data should have required fields
            review_data = data["data"]
            required_fields = [
                "task_id", "decision", "rationale", "citations", 
                "retrieved_doc_ids", "coverage", "latency_ms", 
                "required_actions", "confidence"
            ]
            
            for field in required_fields:
                assert field in review_data, f"Missing field: {field}"
            
            # Decision should be valid
            assert review_data["decision"] in ["approve", "reject"]
            
            # Coverage should be valid
            assert 0.0 <= review_data["coverage"] <= 1.0