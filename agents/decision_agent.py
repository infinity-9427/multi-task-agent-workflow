"""
Decision Agent - Makes approval/rejection decisions using gemini-1.5-flash-2.0
"""

import json
import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):
    """Agent responsible for making task approval/rejection decisions"""
    
    def __init__(self):
        super().__init__()
        self._setup_gemini()
        
    def _setup_gemini(self):
        """Configure Gemini API"""
        genai.configure(api_key=self.settings.GEMINI_API_KEY)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process decision request using gemini-1.5-flash-2.0 with JSON output"""
        try:
            task_details = input_data.get("details", "")
            passages = input_data.get("passages", [])
            tags = input_data.get("tags", [])
            coverage = input_data.get("coverage", 0.0)
            
            if not task_details:
                raise ValueError("details is required")
            
            if not passages or not tags:
                logger.warning("No passages or tags provided")
                return self._create_reject_response()
            
            # Generate LLM decision
            decision_json = await self._generate_decision(task_details, passages, tags)
            
            # Parse and validate JSON
            try:
                decision_data = json.loads(decision_json)
                return self._validate_and_filter_decision(decision_data, tags)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from LLM: {e}")
                return self._create_reject_response()
                
        except Exception as e:
            logger.error(f"Error in DecisionAgent.process: {e}")
            return self._create_reject_response()
    
    async def _generate_decision(self, task_details: str, passages: List[str], tags: List[str]) -> str:
        """Generate decision using gemini-1.5-flash-2.0"""
        try:
            # Truncate context to max chars
            total_context = "\n\n".join(passages)
            if len(total_context) > self.settings.MAX_CONTEXT_CHARS:
                total_context = total_context[:self.settings.MAX_CONTEXT_CHARS]
            
            prompt = f"""Analyze the following task against the provided policy documentation and return a JSON decision.

TASK:
{task_details}

POLICY CONTEXT:
{total_context}

AVAILABLE TAGS:
{json.dumps(tags)}

Return ONLY a valid JSON object with this exact structure:
{{
  "decision": "approve|reject",
  "rationale": "Evidence-based rationale citing specific tags and explaining the decision",
  "citations": ["tag1", "tag2", ...],
  "confidence": 0.0-1.0,
  "required_actions": [
    {{"action": "action_name", "description": "what needs to be done"}},
    ...
  ]
}}

Requirements:
1. Citations MUST only reference tags from the AVAILABLE TAGS list above
2. Use "approve" only if task fully complies with policies and has sufficient citations
3. Use "reject" for non-compliance, insufficient information, or inadequate context
4. Rationale must cite specific tags and explain why approved/rejected
5. Required actions should be specific and actionable for rejected tasks
6. For insufficient information, explain what's missing in the rationale"""

            model = genai.GenerativeModel(
                model_name=self.settings.LLM_MODEL,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000
                )
            )
            
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating decision with LLM: {e}")
            raise
    
    def _validate_and_filter_decision(self, decision_data: Dict[str, Any], available_tags: List[str]) -> Dict[str, Any]:
        """Validate decision JSON and filter citations to subset of available tags"""
        try:
            # Validate required fields
            required_fields = ["decision", "rationale", "citations", "confidence"]
            for field in required_fields:
                if field not in decision_data:
                    logger.error(f"Missing required field: {field}")
                    return self._create_reject_response()
            
            # Validate decision value
            valid_decisions = ["approve", "reject"]
            if decision_data["decision"] not in valid_decisions:
                logger.error(f"Invalid decision: {decision_data['decision']}")
                return self._create_reject_response()
            
            # Filter citations to be subset of available tags
            citations = decision_data.get("citations", [])
            filtered_citations = [tag for tag in citations if tag in available_tags]
            
            # Validate confidence
            confidence = decision_data.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                confidence = 0.0
                
            # Ensure required_actions is properly formatted
            required_actions = decision_data.get("required_actions", [])
            if not isinstance(required_actions, list):
                required_actions = []
                
            return {
                "decision": decision_data["decision"],
                "rationale": decision_data["rationale"],
                "citations": filtered_citations,
                "confidence": float(confidence),
                "required_actions": required_actions
            }
            
        except Exception as e:
            logger.error(f"Error validating decision: {e}")
            return self._create_reject_response()
    
    def _create_reject_response(self) -> Dict[str, Any]:
        """Create standard reject response for processing errors"""
        return {
            "decision": "reject",
            "rationale": "Unable to process decision due to insufficient information or invalid data format",
            "citations": [],
            "confidence": 0.0,
            "required_actions": [
                {
                    "action": "provide_more_context",
                    "description": "Provide additional documentation or context for review"
                }
            ]
        }
    
    def get_agent_type(self) -> str:
        """Return the agent type"""
        return "DecisionAgent"