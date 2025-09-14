import logging
import time
from typing import Dict, Any
from schemas.review import ReviewRequest, ReviewResponse, Decision, Citation, RetrieverResult, DecisionResult
from agents.retriever_agent import RetrieverAgent
from agents.decision_agent import DecisionAgent

logger = logging.getLogger(__name__)


class ReviewRepository:
    _instance = None
    _retriever_agent = None
    _decision_agent = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ReviewRepository, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._retriever_agent is None:
            try:
                self._retriever_agent = RetrieverAgent()
                logger.info("RetrieverAgent initialized at startup")
            except Exception as e:
                logger.error(f"Failed to initialize RetrieverAgent at startup: {str(e)}")
                self._retriever_agent = None
        
        if self._decision_agent is None:
            try:
                self._decision_agent = DecisionAgent()
                logger.info("DecisionAgent initialized at startup")
            except Exception as e:
                logger.error(f"Failed to initialize DecisionAgent at startup: {str(e)}")
                self._decision_agent = None
    
    async def process_review(self, request: ReviewRequest) -> ReviewResponse:
        """
        Process a review request using the multi-agent system.
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing review for task_id: {request.task_id}")
            
            # Validate agents are initialized
            if self._retriever_agent is None:
                raise ValueError("RetrieverAgent not properly initialized")
            if self._decision_agent is None:
                raise ValueError("DecisionAgent not properly initialized")
            
            # Step 1: Use Retriever Agent to get relevant context
            retrieval_input = {
                "task_id": request.task_id,
                "details": request.details
            }
            
            retriever_result = await self._retriever_agent.process(retrieval_input)
            
            # Step 2: Use Decision Agent to make approval/rejection decision
            decision_input = {
                "task_details": request.details,
                "retriever_result": retriever_result
            }
            
            decision_result = await self._decision_agent.process(decision_input)
            
            # Step 3: Validate decision result
            self._validate_decision_result(decision_result, retriever_result)
            
            # Step 4: Create citations for backward compatibility
            citations = []
            for passage in retriever_result.passages:
                try:
                    citation = Citation.from_retrieved_passage(passage)
                    citations.append(citation)
                except Exception as e:
                    logger.warning(f"Failed to create citation from passage: {str(e)}")
                    continue
            
            # Step 5: Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Step 6: Create response using the factory method
            response = ReviewResponse.create_response(
                task_id=request.task_id,
                decision_result=decision_result,
                retriever_result=retriever_result,
                latency_ms=latency_ms,
                citations=citations
            )
            
            logger.info(f"Review processed successfully for task_id: {request.task_id} in {latency_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error processing review: {str(e)}")
            raise
    
    def _validate_decision_result(self, decision_result: DecisionResult, retriever_result: RetrieverResult) -> None:
        """
        Validate decision result according to specification requirements
        """
        try:
            # Validate decision is one of allowed values
            if decision_result.decision not in [Decision.APPROVE, Decision.REJECT, Decision.REJECT_DUE_TO_INSUFFICIENT_CONTEXT]:
                raise ValueError(f"Invalid decision: {decision_result.decision}")
            
            # Validate confidence is within range
            if not (0.0 <= decision_result.confidence <= 1.0):
                raise ValueError(f"Invalid confidence: {decision_result.confidence}")
            
            # Get available tags from retriever result
            available_tags = set(passage.tag for passage in retriever_result.passages)
            cited_tags = set(decision_result.cited_tags)
            
            # Validate all citations are subset of retriever's tags
            if not cited_tags.issubset(available_tags):
                invalid_tags = cited_tags - available_tags
                raise ValueError(f"Invalid cited tags not from retriever: {invalid_tags}")
            
            # For approvals, must include multiple distinct citations and sufficient coverage
            if decision_result.decision == Decision.APPROVE:
                if len(decision_result.cited_tags) < 2:
                    raise ValueError("Approvals must cite at least 2 distinct tags")
                
                # Check coverage threshold (using decision agent's threshold)
                if retriever_result.coverage_score < 0.3:
                    raise ValueError("Approvals must have sufficient coverage score")
            
            # For rejections, must include at least one required action
            if decision_result.decision == Decision.REJECT and not decision_result.required_actions:
                raise ValueError("Rejections must include at least one required action")
            
        except Exception as e:
            logger.error(f"Decision validation failed: {str(e)}")
            raise