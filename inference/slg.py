from langgraph.graph import StateGraph, START, END
from transformers import AutoTokenizer, pipeline
from huggingface_hub import InferenceClient
from huggingface_hub import login
from config import CONFIG
from logging_config import logger
import json
import os
import functools


class SmallLanguageGraph:
    def __init__(self, experts_location):
        self.experts_location = experts_location

    @staticmethod
    def _categorize_task(prompt):
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

    @staticmethod
    def _tuned_generate(prompt, model):
        logger.info("Generating from tuned")
        messages = [
            {"role": "system", "content": CONFIG['inference_prompt']},
            {"role": "user", "content": prompt}
        ]

        model_id = model
        logger.info(f"Model used to infer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
        )

        outputs = pipe(
            messages,
            max_new_tokens=CONFIG['max_new_tokens'],
            temperature=CONFIG['temperature'],
            tokenizer=tokenizer
        )
        logger.info("Inference complete.")

        return outputs[0]["generated_text"][-1]['content']

    # Step 2: Define Node Functions
    def _task_analysis_node(self, state):
        """Analyze the task and route to appropriate expert."""
        question = state["question"]
        experts_list = "\n".join(
            f"- {expert}" for expert in os.listdir(f'experiments/{self.experts_location}'))
        prompt = (
            f"Analyze the question below and find an expert from the following list which can answer the question:\n"
            f"{experts_list}\n"
            f"Question: {question}\n"
            f"Return the category only."
        )
        response = self._categorize_task(prompt)
        state["category"] = response.strip().lower()
        return state

    def _expert_node_builder(self, state, model):
        """Damage classification expert node."""
        question = state["question"]
        prompt = f"""You are an expert in aerospace engineering.
        Provide a detailed answer to the following question: {question}"""
        state["answer"] = self._tuned_generate(
            prompt,
            f"experiments/{self.experts_location}/{model}"
        )
        return state

    def _routing_function(self, state):
        """Route based on the category identified in the task analysis."""
        models = {folder_name: folder_name for folder_name in os.listdir(f'experiments/{self.experts_location}')}
        category = state.get("category")
        return models.get(category, END)

    def _build_graph(self):
        logger.info("Building graph.")
        graph_builder = StateGraph(dict)

        logger.info("Adding nodes to the graph.")
        graph_builder.add_node("task_analysis", self._task_analysis_node)
        for node in os.listdir(f'experiments/{self.experts_location}'):
            if not node.endswith(".json"):
                logger.info(f"Adding node {node}.")
                graph_builder.add_node(
                    node, functools.partial(
                    self._expert_node_builder,
                    model=node
                    )
                )

        logger.info("Adding edges to the graph.")
        graph_builder.add_edge(START, "task_analysis")
        graph_builder.add_conditional_edges("task_analysis", self._routing_function)
        for edge in os.listdir(f'experiments/{self.experts_location}'):
            if not edge.endswith(".json"):
                logger.info(f"Adding edge {edge}.")
                graph_builder.add_edge(edge, END)

        graph = graph_builder.compile()
        logger.info("Graph built.")
        logger.info(40*'-')

        return graph

    # Step 4: Execute the Graph
    def ask_slg(self, file, inference_model):
        """Run the graph for a user question."""
        # Step 1: Read the original JSON file
        with open(file, 'r') as f:
            data = json.load(f)

        # Step 2: Process the data
        graph = self._build_graph()
        answers_list = []
        for item in data:
            logger.info(f"Inference of the title: {item['title']}")
            initial_state = {"question": item['question']}
            result = graph.invoke(initial_state)

            new_dict = {
                "chapter": item['chapter'],
                "title": item['title'],
                "question": item['question'],
                "answer": result.get("answer")
            }
            answers_list.append(new_dict)
            logger.info(40*'-')

        with open(f'answers/{inference_model}.json', 'w') as f:
            json.dump(answers_list, f, indent=4)

        return None
