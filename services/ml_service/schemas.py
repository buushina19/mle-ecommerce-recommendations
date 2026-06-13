from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    visitorid: int
    recommendations: list[int] = Field(..., description="Top item IDs")
    strategy: str
