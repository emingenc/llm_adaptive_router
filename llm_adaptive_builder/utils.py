from typing import Dict, Any
from .models import RouteMetadata

def create_route_metadata(
    model: str,
    invoker: Any,
    capabilities: list[str],
    cost: float = 0.0,
    performance_score: float = 0.0,
    example_sentences: list[str] = [],
    additional_info: Dict[str, Any] = {}
) -> RouteMetadata:
    return RouteMetadata(
        model=model,
        invoker=invoker,
        capabilities=capabilities,
        cost=cost,
        performance_score=performance_score,
        example_sentences=example_sentences,
        additional_info=additional_info
    )
