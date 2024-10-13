# LLM Adaptive Router

LLM Adaptive Router is a Python package that enables dynamic model selection based on query content. It uses efficient vector search for initial categorization and LLM-based fine-grained selection for complex cases. The router can adapt and learn from feedback, making it suitable for a wide range of applications.

## Features

- Dynamic model selection based on query content
- Efficient vector search for initial categorization
- LLM-based fine-grained selection for complex cases
- Adaptive learning from feedback
- Flexible configuration of routes and models
- Easy integration with LangChain and various LLM providers

## Installation

You can install LLM Adaptive Router using pip:

```bash
pip3 install llm-adaptive-router
```

## Quick Start

Here's a basic example of how to use LLM Adaptive Router:

```python
from llm_adaptive_router import AdaptiveRouter, create_route_metadata
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Initialize LLM models
gpt_3_5_turbo = ChatOpenAI("gpt-3.5-turbo", temperature=0)
codex = ChatOpenAI("codex", temperature=0)
gpt_4 = ChatOpenAI("gpt-4", temperature=0)

# Define routes
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

# Initialize router
router = AdaptiveRouter(
    vectorstore=Chroma(embedding_function=OpenAIEmbeddings()),
    llm=ChatOpenAI("gpt-3.5-turbo", temperature=0),
    embeddings=OpenAIEmbeddings(),
    routes=routes
)

# Use the router
query = "Write a Python function to calculate the Fibonacci sequence"
selected_model_route = router.route(query)
selected_model_name = selected_model_route.model
invoker = selected_model_route.invoker
response = invoker.invoke(query)

print(f"Selected model: {selected_model_name}")
print(f"Response: {response}")
```

## Detailed Usage

### Creating Route Metadata

Use the `create_route_metadata` function to define routes:

```python
from llm_adaptive_router import create_route_metadata

route = create_route_metadata(
    model="model_name",
    invoker=model_function,
    capabilities=["capability1", "capability2"],
    cost=0.01,
    example_sentences=["Example query 1", "Example query 2"],
    additional_info={"key": "value"}
)
```

### Initializing the AdaptiveRouter

Create an instance of `AdaptiveRouter` with your configured routes:

```python
router = AdaptiveRouter(
    vectorstore=your_vectorstore,
    llm=your_llm,
    embeddings=your_embeddings,
    routes=your_routes
)
```

### Routing Queries

Use the `route` method to select the appropriate model for a query:

```python
selected_model_route = router.route("Your query here")
selected_model_name = selected_model_route.model
invoker = selected_model_route.invoker
response = invoker.invoke("Your query here")
```

### Adding Feedback

Improve the router's performance by providing feedback:

```python
router.add_feedback(query, selected_model, performance_score)
```

### Advanced Features

- Custom Vector Stores: LLM Adaptive Router supports various vector stores. You can use any vector store that implements the `VectorStore` interface from LangChain.
- Dynamic Route Updates: You can add or remove routes dynamically:

```python
router.add_route("new_route", new_route_metadata)
router.remove_route("old_route")
```

- Adjusting Router Behavior: Fine-tune the router's behavior:

```python
router.set_complexity_threshold(0.8)
router.set_update_frequency(200)
```

