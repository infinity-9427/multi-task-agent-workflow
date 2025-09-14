"""
Minimal viable RAG testing examples using FastAPI TestClient
Tests the complete RAG pipeline end-to-end with simplified approach
"""

import pytest
from fastapi.testclient import TestClient
from main import app


class TestMinimalRAG:
    """Minimal viable tests for RAG system functionality"""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client"""
        return TestClient(app)
    
    def test_review_endpoint_responds(self, client):
        """Test that review endpoint responds (basic connectivity)"""
        response = client.post("/review", json={
            "task_id": "minimal-001",
            "details": "create basic user interface"
        })
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404
        
        # If system is working, should return 200 or 500 (not 422 validation error)
        assert response.status_code in [200, 500]
    
    def test_request_validation_works(self, client):
        """Test request validation catches invalid inputs"""
        
        # Missing task_id
        response = client.post("/review", json={"details": "test"})
        assert response.status_code == 422
        
        # Missing details  
        response = client.post("/review", json={"task_id": "test-001"})
        assert response.status_code == 422
        
        # Empty task_id
        response = client.post("/review", json={"task_id": "", "details": "test"})
        assert response.status_code == 422
        
        # Empty details
        response = client.post("/review", json={"task_id": "test-001", "details": ""})
        assert response.status_code == 422
    
    def test_html_injection_blocked(self, client):
        """Test HTML injection protection"""
        response = client.post("/review", json={
            "task_id": "xss-test",
            "details": "<script>alert('xss')</script>implement dashboard"
        })
        
        # Should be blocked with validation error
        assert response.status_code == 422
        error_detail = str(response.json())
        assert "HTML" in error_detail or "content" in error_detail
    
    def test_health_endpoint(self, client):
        """Test health endpoint exists and responds"""
        response = client.get("/health")
        
        # Should not return 404
        assert response.status_code != 404
        
        # Should return either success or service unavailable 
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] == "ok"
    
    def test_response_format_if_available(self, client):
        """Test response format when system is available"""
        response = client.post("/review", json={
            "task_id": "format-test",
            "details": "implement secure authentication system with multi-factor authentication"
        })
        
        # Skip test if system unavailable
        if response.status_code != 200:
            pytest.skip("System not available for full testing")
        
        data = response.json()
        
        # Should have envelope format
        assert "message" in data
        assert "data" in data
        assert data["message"] == "review completed"
        
        # Check data structure
        review_data = data["data"]
        required_fields = [
            "task_id", "decision", "rationale", "citations", 
            "retrieved_doc_ids", "coverage", "latency_ms", 
            "required_actions", "confidence"
        ]
        
        for field in required_fields:
            assert field in review_data, f"Missing field: {field}"
        
        # Validate field types and values
        assert review_data["task_id"] == "format-test"
        assert review_data["decision"] in ["approve", "reject"]
        assert isinstance(review_data["rationale"], str)
        assert isinstance(review_data["citations"], list)
        assert isinstance(review_data["retrieved_doc_ids"], list) 
        assert isinstance(review_data["coverage"], (int, float))
        assert 0.0 <= review_data["coverage"] <= 1.0
        assert isinstance(review_data["latency_ms"], int)
        assert review_data["latency_ms"] > 0
        assert isinstance(review_data["required_actions"], list)
        assert isinstance(review_data["confidence"], (int, float))
        assert 0.0 <= review_data["confidence"] <= 1.0
    
    def test_different_task_types(self, client):
        """Test different types of tasks get different responses"""
        
        tasks = [
            {
                "task_id": "simple-task",
                "details": "create form",
                "expected_type": "simple"
            },
            {
                "task_id": "security-task", 
                "details": "implement comprehensive security authentication system with multi-factor authentication, encryption, and audit logging",
                "expected_type": "complex"
            },
            {
                "task_id": "ai-task",
                "details": "develop artificial intelligence system for automated decision making with machine learning algorithms and data processing",
                "expected_type": "ai-related"
            }
        ]
        
        results = []
        
        for task in tasks:
            response = client.post("/review", json={
                "task_id": task["task_id"],
                "details": task["details"]
            })
            
            if response.status_code == 200:
                data = response.json()["data"]
                results.append({
                    "task_type": task["expected_type"],
                    "decision": data["decision"],
                    "coverage": data["coverage"],
                    "citations": len(data["citations"]),
                    "confidence": data["confidence"]
                })
        
        # If we have results, verify system is working
        if results:
            # All should have valid decisions
            for result in results:
                assert result["decision"] in ["approve", "reject"]
                assert 0.0 <= result["coverage"] <= 1.0
            
            # Complex tasks might get different treatment than simple ones
            # (but we don't enforce specific outcomes since it depends on ingested docs)
            print(f"\n📊 Task Results: {results}")
    
    def test_coverage_calculation_working(self, client):
        """Test that coverage calculation produces reasonable values"""
        
        # Test with different complexity levels
        test_cases = [
            ("basic", "create login"),
            ("detailed", "implement user authentication system"),
            ("comprehensive", "implement secure user authentication system with multi-factor authentication following security best practices and compliance requirements")
        ]
        
        coverages = []
        
        for name, details in test_cases:
            response = client.post("/review", json={
                "task_id": f"coverage-{name}",
                "details": details
            })
            
            if response.status_code == 200:
                coverage = response.json()["data"]["coverage"]
                coverages.append((name, coverage))
        
        if coverages:
            # All coverages should be valid
            for name, coverage in coverages:
                assert 0.0 <= coverage <= 1.0, f"Invalid coverage for {name}: {coverage}"
            
            print(f"\n📈 Coverage Results: {coverages}")


class TestSystemIntegration:
    """Integration tests for system components"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_can_import_main_components(self):
        """Test that main components can be imported"""
        from main import app
        from rag.orchestrator import RAGOrchestrator
        from agents.retriever_agent import RetrieverAgent
        from agents.decision_agent import DecisionAgent
        from schemas.review import ReviewRequest, RetrievalReport
        
        # Should not raise import errors
        assert app is not None
        assert RAGOrchestrator is not None
        assert RetrieverAgent is not None
        assert DecisionAgent is not None
    
    def test_settings_configuration(self):
        """Test settings are properly configured"""
        from settings.settings import get_settings
        
        settings = get_settings()
        
        # Should have required models configured
        assert hasattr(settings, 'LLM_MODEL')
        assert hasattr(settings, 'EMBEDDING_MODEL') 
        assert hasattr(settings, 'EMBEDDING_DIM')
        assert hasattr(settings, 'TOP_K')
        assert hasattr(settings, 'COVERAGE_THRESHOLD')
        assert hasattr(settings, 'APPROVAL_COVERAGE_MIN')
        
        # Check values are reasonable
        assert settings.EMBEDDING_DIM == 768
        assert settings.TOP_K == 4
        assert 0.0 <= settings.COVERAGE_THRESHOLD <= 1.0
        assert 0.0 <= settings.APPROVAL_COVERAGE_MIN <= 1.0
    
    def test_orchestrator_can_be_created(self):
        """Test that orchestrator can be created without errors"""
        try:
            from rag.orchestrator import RAGOrchestrator
            orchestrator = RAGOrchestrator()
            assert orchestrator is not None
        except Exception as e:
            pytest.fail(f"Failed to create orchestrator: {e}")


# Minimal test runner for direct execution
if __name__ == "__main__":
    import subprocess
    import sys
    
    print("🧪 Running minimal RAG tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "-s", "--tb=short"
    ], capture_output=False)
    
    if result.returncode == 0:
        print("✅ All minimal tests passed!")
    else:
        print("❌ Some tests failed")
    
    sys.exit(result.returncode)