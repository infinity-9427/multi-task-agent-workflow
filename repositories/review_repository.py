from schemas.review import ReviewRequest, ReviewResponse


class ReviewRepository:
    def process_review(self, request: ReviewRequest) -> ReviewResponse:
        """
        Process a review request and return the response.
        """
        return ReviewResponse.create_response(request.details)