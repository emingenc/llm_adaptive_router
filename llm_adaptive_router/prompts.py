router_prompt_template = """
You are an assistant that determines the most appropriate route based on the user's question.

Question:
{query}

Available Routes:
{routes}

Choose the best route from the available options.

Respond **only** in the following JSON format:

{{
    "route": "selected_model",
    "confidence": confidence_score_between_0.0_and_1.0
}}

{format_instructions}
"""