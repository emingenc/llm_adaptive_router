from pydantic import BaseModel, Field
from typing import List, Callable, Any, Dict, Optional


class RouteMetadata(BaseModel):
    invoker: Callable = Field(..., exclude=True)  # Required field
    name: Optional[str] = None
    capabilities: Optional[List[str]] = None
    cost: Optional[float] = None
    example_sentences: Optional[List[str]] = None
    additional_info: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Callable: lambda v: v.__name__,
            list: lambda v: ", ".join(str(i) for i in v) if v else "",
            dict: lambda v: str(v) if v else "",
        }
        extra = "allow"


class QueryResult(BaseModel):
    content: str
    metadata: RouteMetadata


class RouteSelection(BaseModel):
    route: str = Field(description="The selected route. best llm model")
    confidence: float = Field(
        description="Confidence score for the chosen route (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
