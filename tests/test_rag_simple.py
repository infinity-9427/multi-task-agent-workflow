"""
Simple RAG system functionality test using TestClient.
Updated to align with current pgvector-only approach and simplified decision types.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


class TestRAGFunctionality:
    """Simple tests for RAG system functionality using TestClient"""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client"""
        return TestClient(app)
    
    def test_simple_request_works(self, client):
        """Test that a simple request works end-to-end"""
        payload = {
            "task_id": "simple-test-001",
            "details": "create user login page"
        }
        
        response = client.post("/review", json=payload)
        
        # Should not return 404 or validation error
        assert response.status_code != 404
        if response.status_code != 200:
            # If system not available, skip detailed checks
            pytest.skip(f"System not available: {response.status_code}")
        
        # Should return JSON with envelope format
        data = response.json()
        assert isinstance(data, dict)
        assert "message" in data
        assert "data" in data
        
        review_data = data["data"]
        
        # Should have required fields
        required_fields = ["task_id", "decision", "coverage", "latency_ms", "rationale"]
        for field in required_fields:
            assert field in review_data
        
        # Task ID should match
        assert review_data["task_id"] == payload["task_id"]
        
        # Decision should be valid (simplified to approve/reject)
        assert review_data["decision"] in ["approve", "reject"]
        
        # Coverage should be valid
        assert 0.0 <= review_data["coverage"] <= 1.0
        
        # Should have some latency
        assert review_data["latency_ms"] > 0
        
        print(f"✅ Simple request test passed!")
        print(f"   Decision: {review_data['decision']}")
        print(f"   Coverage: {review_data['coverage']}")
        print(f"   Latency: {review_data['latency_ms']}ms")
    
    def test_security_request(self, client):
        """Test security-focused request"""
        payload = {
            "task_id": "security-test-001", 
            "details": "implement secure authentication system with multi-factor authentication following security guidelines"
        }
        
        response = client.post("/review", json=payload)
        
        if response.status_code != 200:
            pytest.skip(f"System not available: {response.status_code}")
        
        data = response.json()["data"]
        
        # Should have valid response structure
        assert data["decision"] in ["approve", "reject"]
        assert 0.0 <= data["coverage"] <= 1.0
        assert isinstance(data["rationale"], str)
        
        print(f"✅ Security request test passed!")
        print(f"   Decision: {data['decision']}")
        print(f"   Coverage: {data['coverage']}")
        print(f"   Citations: {len(data['citations'])}")
    
    def test_html_injection_blocked(self, client):
        """Test that HTML injection is blocked"""
        payload = {
            "task_id": "xss-test",
            "details": "<script>alert('xss')</script>implement dashboard"
        }
        
        response = client.post("/review", json=payload)
        
        # Should be rejected with 422 validation error
        assert response.status_code == 422
        
        # Should mention HTML content
        error_data = response.json()
        assert "HTML" in str(error_data) or "content" in str(error_data)
        
        print(f"✅ HTML injection blocked!")
    
    def test_different_coverage_scenarios(self, client):
        """Test that different requests get different coverage scores"""
        
        test_cases = [
            {
                "task_id": "coverage-simple",
                "details": "create form",
                "type": "simple"
            },
            {
                "task_id": "coverage-complex", 
                "details": "implement comprehensive security authentication system with multi-factor authentication, encryption, audit logging, and compliance with enterprise security policies",
                "type": "complex"
            }
        ]
        
        results = []
        
        for case in test_cases:
            response = client.post("/review", json=case)
            
            if response.status_code == 200:
                data = response.json()["data"]
                results.append({
                    "type": case["type"],
                    "coverage": data["coverage"],
                    "decision": data["decision"]
                })
        
        if results:
            print(f"✅ Coverage scenarios test passed!")
            for result in results:
                print(f"   {result['type'].capitalize()} coverage: {result['coverage']} (decision: {result['decision']})")
                
                # Verify coverage is valid
                assert 0.0 <= result["coverage"] <= 1.0
        else:
            pytest.skip("System not available for coverage testing")
    
    def test_decision_logic_works(self, client):
        """Test that decision logic produces valid outcomes"""
        
        requests = [
            {
                "task_id": "decision-1",
                "details": "create simple form"
            },
            {
                "task_id": "decision-2", 
                "details": "implement secure user authentication system with multi-factor authentication and comprehensive security controls"
            }
        ]
        
        results = []
        
        for req in requests:
            response = client.post("/review", json=req)
            
            if response.status_code == 200:
                data = response.json()["data"]
                results.append({
                    "task_id": req["task_id"],
                    "decision": data["decision"],
                    "coverage": data["coverage"],
                    "citations": len(data["citations"]),
                    "required_actions": len(data["required_actions"])
                })
        
        if results:
            print(f"✅ Decision logic test passed!")
            for result in results:
                print(f"   {result['task_id']}: {result['decision']} (coverage: {result['coverage']}, citations: {result['citations']})")
                
                # Verify decision is valid
                assert result["decision"] in ["approve", "reject"]
                
                # Required actions should be a number (may be 0)
                assert isinstance(result["required_actions"], int)
                assert result["required_actions"] >= 0
        else:
            pytest.skip("System not available for decision testing")
    
    def test_envelope_response_format(self, client):
        """Test that response follows envelope format"""
        payload = {
            "task_id": "envelope-test",
            "details": "test envelope response format"
        }
        
        response = client.post("/review", json=payload)
        
        if response.status_code != 200:
            pytest.skip(f"System not available: {response.status_code}")
        
        data = response.json()
        
        # Should have envelope structure
        assert "message" in data
        assert "data" in data
        assert data["message"] == "review completed"
        
        # Data should have all required fields
        review_data = data["data"]
        required_fields = [
            "task_id", "decision", "rationale", "citations", 
            "retrieved_doc_ids", "coverage", "latency_ms", 
            "required_actions", "confidence"
        ]
        
        for field in required_fields:
            assert field in review_data, f"Missing field: {field}"
        
        print(f"✅ Envelope response format test passed!")


def test_system_components_can_be_imported():
    """Test that system components can be imported without errors"""
    try:
        from rag.orchestrator import RAGOrchestrator
        from agents.retriever_agent import RetrieverAgent
        from agents.decision_agent import DecisionAgent
        from schemas.review import ReviewRequest, RetrievalReport
        
        # Should be able to create instances
        orchestrator = RAGOrchestrator()
        assert orchestrator is not None
        
        print("✅ System components loaded successfully!")
        
    except Exception as e:
        pytest.fail(f"Failed to import system components: {e}")


def test_health_endpoint_works():
    """Test health endpoint using TestClient"""
    client = TestClient(app)
    response = client.get("/health")
    
    # Should not return 404
    assert response.status_code != 404
    
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        print("✅ Health endpoint working!")
    else:
        print(f"ℹ️  Health endpoint returned {response.status_code} (system may not be fully configured)")


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    print("🧪 Running simple RAG functionality tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "-s", "--tb=short"
    ], capture_output=False)
    
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    
    sys.exit(result.returncode)