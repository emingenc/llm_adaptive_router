from llm_adaptive_router import AdaptiveRouter, create_route_metadata
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

gpt_3_5_turbo = ChatOpenAI("gpt-3.5-turbo", temperature=0)
codex = ChatOpenAI("codex", temperature=0)
gpt_4 = ChatOpenAI("gpt-4o", temperature=0)

routes = {
    "general": create_route_metadata(
        model="gpt-3.5-turbo",
        invoker=gpt_3_5_turbo,
        capabilities=["general knowledge"],
        cost=0.002,
        example_sentences=["What is the capital of France?", "Explain photosynthesis."]
    ),
    "code": create_route_metadata(
        model="codex",
        invoker=codex,
        capabilities=["code generation", "debugging"],
        cost=0.005,
        example_sentences=["Write a Python function to sort a list.", "Debug this JavaScript code."]
    ),
    "math": create_route_metadata(
        model="gpt-4",
        invoker=gpt_4,
        capabilities=["advanced math", "problem solving"],
        cost=0.01,
        example_sentences=["Solve this differential equation.", "Prove the Pythagorean theorem."]
    )
}

llm = ChatOpenAI("gpt-3.5-turbo", temperature=0)

router = AdaptiveRouter(
    vectorstore=Chroma(embedding_function=OpenAIEmbeddings()),
    llm=llm,
    embeddings=OpenAIEmbeddings(),
    routes=routes
)

query = "Write a Python function to calculate the Fibonacci sequence"
selected_model_route = router.route(query)
selected_model_name = selected_model_route.model
invoker = selected_model_route.invoker
response = invoker.invoke(query)

print(f"Response: {response}")