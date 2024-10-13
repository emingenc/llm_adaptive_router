import pytest
from llm_adaptive_router import AdaptiveRouter, RouteMetadata
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

gpt_3_5_turbo = ChatOpenAI("gpt-3.5-turbo", temperature=0)
codex = ChatOpenAI("codex", temperature=0)
gpt_4 = ChatOpenAI("gpt-4o", temperature=0)


@pytest.fixture
def sample_router():
    routes = {
        "general": RouteMetadata(
            model="gpt-3.5-turbo",
            invoker=gpt_3_5_turbo,
            capabilities=["general knowledge"],
            cost=0.002,
        ),
        "code": RouteMetadata(
            model="codex",
            invoker=codex,
            capabilities=["code generation", "debugging"],
            cost=0.005,
        ),
        "math": RouteMetadata(
            model="gpt-4",
            invoker=gpt_4,
            capabilities=["advanced math", "problem solving"],
            cost=0.01,
        ),
    }
    return AdaptiveRouter(
        vectorstore=Chroma(embedding_function=OpenAIEmbeddings()),
        llm=ChatOpenAI(),
        embeddings=OpenAIEmbeddings(),
        routes=routes,
    )


def test_router_initialization(sample_router):
    assert isinstance(sample_router, AdaptiveRouter)
    assert len(sample_router.get_routes()) == 3


def test_routing(sample_router):
    query = "What is the capital of France?"
    result = sample_router.route(query)
    assert result.model in ["gpt-3.5-turbo", "codex", "gpt-4"]


# Add more tests as needed
