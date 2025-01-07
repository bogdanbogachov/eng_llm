from langgraph.graph import StateGraph, START, END
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient
from huggingface_hub import login
from logging_config import setup_logger
import yaml


# Load config parameters
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize the logger
logger = setup_logger()


# Step 1: Load LLaMA Model and Tokenizer
api_key = config['api_key']
model_id = config['model_id']
inference_prompt = config['inference_prompt']
max_new_tokens = config['max_new_tokens']
seed = config['seed']
temperature = config['temperature']
tokenizer = AutoTokenizer.from_pretrained(model_id)

login(api_key)
client = InferenceClient()

# Function to generate responses using the LLaMA model
def generate_response(prompt):
    messages = [
        {"role": "system", "content": inference_prompt},
        {"role": "user", "content": prompt}
    ]

    total = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    llm_response = client.text_generation(
        total,
        model=model_id,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature
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
    print(response)
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
        print('mech')
        return "mechanical"
    elif category == "electrical":
        print('electro')
        return "electrical"
    elif category in ["general", "software"]:
        print('general')
        return "general"
    else:
        print('oops')
        return END

graph_builder.add_edge(START, "task_analysis")
graph_builder.add_conditional_edges("task_analysis", routing_function)
graph_builder.add_edge("mechanical", END)
graph_builder.add_edge("electrical", END)
graph_builder.add_edge("general", END)

# Compile the graph
graph = graph_builder.compile()

# Step 4: Execute the Graph
def ask_engineering_question(question):
    """Run the graph for a user question."""
    initial_state = {"question": question}
    print('initial state set')
    result = graph.invoke(initial_state)
    return result.get("answer")
