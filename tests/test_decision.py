"""
Tests for DecisionAgent approval requirements
"""

import pytest
from unittest.mock import AsyncMock, patch
from agents.decision_agent import DecisionAgent


class TestDecisionAgent:
    """Test DecisionAgent approval logic and policy gates"""
    
    @pytest.fixture
    def decision_agent(self):
        return DecisionAgent()
    
    @pytest.mark.asyncio  
    async def test_approval_requires_two_distinct_citations(self, decision_agent):
        """Test that approval requires >= 2 distinct citations"""
        input_data = {
            "details": "test task",
            "passages": ["passage1", "passage2", "passage3"],
            "tags": ["doc:1#chunk:1", "doc:2#chunk:2", "doc:1#chunk:3"],
            "coverage": 0.8
        }
        
        # Mock LLM to return approval with 2+ distinct citations
        mock_json = """{
            "decision": "approve",
            "rationale": "Task meets requirements based on doc:1#chunk:1 and doc:2#chunk:2",
            "citations": ["doc:1#chunk:1", "doc:2#chunk:2"],
            "confidence": 0.9,
            "required_actions": []
        }"""
        
        with patch.object(decision_agent, '_generate_decision') as mock_generate:
            mock_generate.return_value = mock_json
            
            result = await decision_agent.process(input_data)
            
            # Should approve since we have 2 distinct citations
            assert result["decision"] == "approve"
            assert len(result["citations"]) == 2
    
    @pytest.mark.asyncio
    async def test_citations_filtered_to_available_tags(self, decision_agent):
        """Test that citations are filtered to be subset of available tags"""
        input_data = {
            "details": "test task",
            "passages": ["passage1", "passage2"],
            "tags": ["doc:1#chunk:1", "doc:2#chunk:2"],
            "coverage": 0.8
        }
        
        # Mock LLM to return citations including invalid ones
        mock_json = """{
            "decision": "approve",
            "rationale": "Test rationale",
            "citations": ["doc:1#chunk:1", "doc:2#chunk:2", "doc:3#chunk:99"],
            "confidence": 0.9,
            "required_actions": []
        }"""
        
        with patch.object(decision_agent, '_generate_decision') as mock_generate:
            mock_generate.return_value = mock_json
            
            result = await decision_agent.process(input_data)
            
            # Should only include valid citations that exist in available tags
            assert "doc:3#chunk:99" not in result["citations"]
            assert set(result["citations"]).issubset(set(input_data["tags"]))
    
    @pytest.mark.asyncio
    async def test_invalid_json_returns_reject(self, decision_agent):
        """Test that invalid JSON from LLM returns reject"""
        input_data = {
            "details": "test task",
            "passages": ["passage1"],
            "tags": ["doc:1#chunk:1"],
            "coverage": 0.8
        }
        
        # Mock LLM to return invalid JSON
        with patch.object(decision_agent, '_generate_decision') as mock_generate:
            mock_generate.return_value = "invalid json response"
            
            result = await decision_agent.process(input_data)
            
            assert result["decision"] == "reject"
            assert result["citations"] == []
            assert result["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_missing_required_fields_returns_reject(self, decision_agent):
        """Test that missing required fields returns reject"""
        input_data = {
            "details": "test task", 
            "passages": ["passage1"],
            "tags": ["doc:1#chunk:1"],
            "coverage": 0.8
        }
        
        # Mock LLM to return JSON missing required fields
        mock_json = """{
            "decision": "approve"
        }"""
        
        with patch.object(decision_agent, '_generate_decision') as mock_generate:
            mock_generate.return_value = mock_json
            
            result = await decision_agent.process(input_data)
            
            assert result["decision"] == "reject"
    
    @pytest.mark.asyncio
    async def test_empty_passages_returns_reject(self, decision_agent):
        """Test that empty passages/tags returns reject immediately"""
        input_data = {
            "details": "test task",
            "passages": [],
            "tags": [],
            "coverage": 0.0
        }
        
        result = await decision_agent.process(input_data)
        
        assert result["decision"] == "reject"
        assert result["citations"] == []
        assert result["confidence"] == 0.0