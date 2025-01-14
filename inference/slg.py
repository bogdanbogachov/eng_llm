from langgraph.graph import StateGraph, START, END
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient
from huggingface_hub import login
from config import CONFIG
from logging_config import logger
import json


def generate_response(prompt):
    """
    Function to generate responses using the LLaMA model
    """
    messages = [
        {"role": "system", "content": CONFIG['inference_prompt']},
        {"role": "user", "content": prompt}
    ]

    login(CONFIG['api_key'])
    client = InferenceClient()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_id'])
    total = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    llm_response = client.text_generation(
        total,
        model=CONFIG['model_id'],
        max_new_tokens=CONFIG['max_new_tokens'],
        seed=CONFIG['seed'],
        temperature=CONFIG['temperature']
    )
    return llm_response

# Step 2: Define Node Functions
def task_analysis_node(state):
    """Analyze the task and route to appropriate expert."""
    question = state["question"]
    prompt = (
        f"Analyze the following question and categorize it as one of the following: "
        f"'mechanical', 'electrical', 'software', or 'general'. Question: {question}"
        f"Return the category only."
    )
    response = generate_response(prompt)
    state["category"] = response.strip().lower()
    return state

def mechanical_expert_node(state):
    """Mechanical expert node."""
    question = state["question"]
    prompt = f"You are an expert in mechanical engineering. Provide a detailed answer to the following question: {question}"
    state["answer"] = generate_response(prompt)
    return state

def electrical_expert_node(state):
    """Electrical expert node."""
    question = state["question"]
    prompt = f"You are an expert in electrical engineering. Provide a detailed answer to the following question: {question}"
    state["answer"] = generate_response(prompt)
    return state

def general_expert_node(state):
    """General expert node."""
    question = state["question"]
    prompt = f"Provide a general engineering response to the following question: {question}"
    state["answer"] = generate_response(prompt)
    return state

# Step 3: Create the Graph
graph_builder = StateGraph(dict)

# Add nodes to the graph
graph_builder.add_node("task_analysis", task_analysis_node)
graph_builder.add_node("mechanical", mechanical_expert_node)
graph_builder.add_node("electrical", electrical_expert_node)
graph_builder.add_node("general", general_expert_node)

# Define transitions between nodes
def routing_function(state):
    """Route based on the category identified in the task analysis."""
    category = state.get("category")
    if category == "mechanical":
        return "mechanical"
    elif category == "electrical":
        return "electrical"
    elif category in ["general", "software"]:
        return "general"
    else:
        return END

graph_builder.add_edge(START, "task_analysis")
graph_builder.add_conditional_edges("task_analysis", routing_function)
graph_builder.add_edge("mechanical", END)
graph_builder.add_edge("electrical", END)
graph_builder.add_edge("general", END)

# Compile the graph
graph = graph_builder.compile()

# Step 4: Execute the Graph
def ask_slg(file, inference_model):
    """Run the graph for a user question."""
    # Step 1: Read the original JSON file
    with open(file, 'r') as f:
        data = json.load(f)

    # Step 2: Process the data
    answers_list = []
    for item in data:
        initial_state = {"question": item['question']}
        result = graph.invoke(initial_state)

        new_dict = {
            "chapter": item['chapter'],
            "title": item['title'],
            "question": item['question'],
            "answer": result.get("answer")
        }
        answers_list.append(new_dict)

    with open(f'answers/{inference_model}.json', 'w') as f:
        json.dump(answers_list, f, indent=4)

    return None
