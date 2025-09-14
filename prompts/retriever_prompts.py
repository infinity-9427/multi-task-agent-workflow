"""
Prompts for the Retriever Agent
"""

DOCUMENT_ANALYSIS_PROMPT = """
You are a document analysis expert. Your task is to analyze the retrieved document chunks and extract the most relevant information for task review.

Retrieved document chunks:
{context}

Task being reviewed:
Task ID: {task_id}
Details: {details}

Please provide:
1. A relevance assessment (0.0-1.0 scale)
2. Key information that relates to the task
3. Any policy guidelines, requirements, or constraints mentioned
4. Potential risks or considerations

Format your response as structured information that can be used for decision making.
"""

RETRIEVAL_QUERY_PROMPT = """
Convert the following task review request into an optimal search query for retrieving relevant documentation:

Task ID: {task_id}
Task Details: {details}

Generate 2-3 search queries that would help find relevant policies, guidelines, or documentation for reviewing this task.
Focus on key concepts, requirements, and potential compliance issues.

Return only the search queries, one per line.
"""