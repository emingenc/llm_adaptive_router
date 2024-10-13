from llm_adaptive_router import AdaptiveRouter, RouteMetadata
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

gpt_3_5_turbo = ChatOpenAI(model="gpt-3.5-turbo")
mini = ChatOpenAI(model="gpt-4o-mini")
gpt_4 = ChatOpenAI(model="gpt-4")

routes = {
    "general": RouteMetadata(
        invoker=gpt_3_5_turbo,
        capabilities=["general knowledge"],
        cost=0.002,
        example_sentences=["What is the capital of France?", "Explain photosynthesis."]
    ),
    "mini": RouteMetadata(
        invoker=mini,
        capabilities=["general knowledge"],
        cost=0.002,
        example_sentences=["What is the capital of France?", "Explain photosynthesis."]
        
    ),
    "math": RouteMetadata(
        invoker=gpt_4,
        capabilities=["advanced math", "problem solving"],
        cost=0.01,
        example_sentences=["Solve this differential equation.", "Prove the Pythagorean theorem."]
    )
}

llm = ChatOpenAI(model="gpt-3.5-turbo")

router = AdaptiveRouter(
    vectorstore=Chroma(embedding_function=OpenAIEmbeddings()),
    llm=llm,
    embeddings=OpenAIEmbeddings(),
    routes=routes
)

query = "How are you"
query2 = "Write a Python function to hello world"
selected_model_route = router.route(query)
selected_model_name = selected_model_route
print(selected_model_name)
invoker = selected_model_route.invoker
response = invoker.invoke(query)

print(f"Response: {response}")