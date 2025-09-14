"""
RAG Orchestrator with proper flow: retrieve → coverage gate → decide → policy gate → finalize
"""

import logging
import time
from typing import Dict, Any

from agents.retriever_agent import RetrieverAgent
from agents.decision_agent import DecisionAgent
from schemas.review import RetrievalReport, Decision
from settings.settings import get_settings

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Orchestrates the RAG pipeline with proper gating"""
    
    def __init__(self):
        self.settings = get_settings()
        self.retriever = RetrieverAgent()
        self.decision_agent = DecisionAgent()
        
    async def process_review(self, task_id: str, details: str) -> Dict[str, Any]:
        """
        Main orchestration flow:
        1. retrieve → 2. coverage gate → 3. decide → 4. policy gate → 5. finalize
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve
            logger.info(f"Starting retrieval for task {task_id}")
            retrieval_report = await self.retriever.process({
                "task_id": task_id,
                "details": details
            })
            
            # Step 2: Coverage gate
            if retrieval_report.coverage < self.settings.COVERAGE_THRESHOLD:
                logger.warning(f"Coverage {retrieval_report.coverage} below threshold {self.settings.COVERAGE_THRESHOLD}")
                return self._create_low_coverage_response(
                    task_id, retrieval_report, start_time
                )
            
            # Step 3: Decision (only if coverage gate passed)
            logger.info(f"Coverage gate passed, calling DecisionAgent")
            decision_input = {
                "task_id": task_id,
                "details": details,
                "passages": retrieval_report.passages,
                "tags": retrieval_report.tags,
                "coverage": retrieval_report.coverage
            }
            
            decision_result = await self.decision_agent.process(decision_input)
            
            # Step 4: Policy gate
            decision_result = self._apply_policy_gate(decision_result, retrieval_report)
            
            # Step 5: Finalize response
            return self._create_final_response(
                task_id, decision_result, retrieval_report, start_time
            )
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            return self._create_error_response(task_id, str(e), start_time)
            
    def _apply_policy_gate(self, decision_result: Dict[str, Any], retrieval_report: RetrievalReport) -> Dict[str, Any]:
        """Apply policy gate: requires coverage >= 0.45 and >= 2 distinct citations"""
        if decision_result["decision"] == "approve":
            # Check policy requirements
            coverage_ok = retrieval_report.coverage >= self.settings.APPROVAL_COVERAGE_MIN
            citations_ok = len(set(decision_result.get("citations", []))) >= 2
            
            if not (coverage_ok and citations_ok):
                logger.warning(f"Policy gate failed: coverage={retrieval_report.coverage}, citations={len(set(decision_result.get('citations', [])))}")
                decision_result.update({
                    "decision": "reject",
                    "rationale": "Insufficient context or citations for approval. Requires higher coverage and at least 2 distinct citations.",
                    "citations": [],
                    "confidence": 0.0
                })
                
        return decision_result
        
    def _create_low_coverage_response(self, task_id: str, retrieval_report: RetrievalReport, start_time: float) -> Dict[str, Any]:
        """Create response for low coverage"""
        return {
            "message": "review completed",
            "data": {
                "task_id": task_id,
                "decision": "reject",
                "rationale": f"Insufficient contextual information available for review. Coverage {retrieval_report.coverage:.2f} is below required threshold {self.settings.COVERAGE_THRESHOLD}.",
                "citations": [],
                "retrieved_doc_ids": retrieval_report.doc_ids,
                "coverage": retrieval_report.coverage,
                "latency_ms": int((time.time() - start_time) * 1000),
                "required_actions": [
                    {
                        "action": "provide_more_context",
                        "description": "Provide additional context or documentation for review"
                    }
                ],
                "confidence": 0.0
            }
        }
        
    def _create_final_response(self, task_id: str, decision_result: Dict[str, Any], retrieval_report: RetrievalReport, start_time: float) -> Dict[str, Any]:
        """Create final orchestrated response"""
        return {
            "message": "review completed",
            "data": {
                "task_id": task_id,
                "decision": decision_result["decision"],
                "rationale": decision_result["rationale"],
                "citations": decision_result.get("citations", []),
                "retrieved_doc_ids": retrieval_report.doc_ids,
                "coverage": retrieval_report.coverage,
                "latency_ms": int((time.time() - start_time) * 1000),
                "required_actions": decision_result.get("required_actions", []),
                "confidence": decision_result.get("confidence", 0.0)
            }
        }
        
    def _create_error_response(self, task_id: str, error_msg: str, start_time: float) -> Dict[str, Any]:
        """Create error response"""
        return {
            "message": "review failed",
            "data": {
                "task_id": task_id,
                "decision": "reject",
                "rationale": f"Processing error: {error_msg}. Unable to complete review.",
                "citations": [],
                "retrieved_doc_ids": [],
                "coverage": 0.0,
                "latency_ms": int((time.time() - start_time) * 1000),
                "required_actions": [
                    {
                        "action": "retry_request",
                        "description": "Retry the request after checking system status"
                    }
                ],
                "confidence": 0.0
            }
        }