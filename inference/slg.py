from langgraph.graph import StateGraph, START, END
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from config import CONFIG
from logging_config import logger
from peft import PeftModel
import json
import torch
import os
import functools
import difflib


class SmallLanguageGraph:
    def __init__(self, experts_location, experiment):
        self.experts_location = experts_location
        self.experiment = experiment

    def _categorize_task(self, prompt, experts):
        """
        Function to orchestrate questions.
        """
        messages = [
            {"role": "system", "content": CONFIG['inference_prompt']},
            {"role": "user", "content": prompt}
        ]

        # Set the paths for your local model and adapter
        base_model_path = 'downloaded_3_1_8b'
        adapter_path = f"experiments/{self.experts_location}/finetuned_orchestrator_3_1_8b"

        # Load the tokenizer (from base model)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        # Load the base model from local storage
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,  # Uses FP16 for lower memory usage
            device_map="auto"  # Ensures it loads to GPU automatically
        )

        # Apply the LoRA adapter on top
        model = PeftModel.from_pretrained(model, adapter_path)

        # Ensure the model is fully on GPU
        model.to("cuda")

        # Create the pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )

        outputs = pipe(
            messages,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"]
        )

        output = outputs[0]["generated_text"][-1]['content']

        logger.debug(40*'-')
        logger.debug(40*'-')
        logger.debug(f"Categorizer output: {outputs}")
        logger.debug(40*'-')
        logger.debug(40*'-')

        if output in experts:
            return output
        else:
            closest_match = max(experts, key=lambda s: difflib.SequenceMatcher(None, output, s).ratio())
            return closest_match

    @staticmethod
    def _tuned_generate(prompt, adapter):
        logger.info("Generating from tuned")
        messages = [
            {"role": "system", "content": CONFIG['inference_prompt']},
            {"role": "user", "content": prompt}
        ]

        # Set the paths for your local model and adapter
        base_model_path = 'downloaded_3_2_1b'
        adapter_path = adapter
        logger.info(f"Model used to infer: {adapter}")

        # Load the tokenizer (from base model)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        # Load the base model from local storage
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,  # Uses FP16 for lower memory usage
            device_map="auto"  # Ensures it loads to GPU automatically
        )

        # Apply the LoRA adapter on top
        model = PeftModel.from_pretrained(model, adapter_path)

        # Ensure the model is fully on GPU
        model.to("cuda")

        # Create the pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )

        outputs = pipe(
            messages,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"]
        )
        logger.debug(f"Output: {outputs}")

        logger.info("Inference complete.")

        del model
        del tokenizer
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Helps defragment GPU memory

        return outputs[0]["generated_text"][-1]['content']

    # Step 2: Define Node Functions
    def _task_analysis_node(self, state):
        """Analyze the task and route to appropriate expert."""
        question = state["question"]
        experts_list = "\n".join(
            f"- {expert}" for expert in os.listdir(f'experiments/{self.experts_location}/slg'))
        prompt = (
            f"Analyze this question and find an appropriate expert who can answer it:\n {question}"
            f"A friendly reminder, experts are as follows:\n {experts_list}"
            f"Very important, return only an expert name, nothing else!"
        )
        response = self._categorize_task(prompt, experts_list)
        state["category"] = response.strip().lower()
        return state

    def _expert_node_builder(self, state, model):
        """Damage classification expert node."""
        question = state["question"]
        prompt = f"""You are an expert in aerospace engineering.
        Provide a detailed answer to the following question: {question}"""
        state["answer"] = self._tuned_generate(
            prompt,
            f"experiments/{self.experts_location}/slg/{model}"
        )
        return state

    def _routing_function(self, state):
        """Route based on the category identified in the task analysis."""
        models = {folder_name: folder_name for folder_name in os.listdir(f'experiments/{self.experts_location}/slg')}
        category = state.get("category")
        return models.get(category, END)

    def _build_graph(self):
        logger.info("Building graph.")
        graph_builder = StateGraph(dict)

        logger.info("Adding nodes to the graph.")
        graph_builder.add_node("task_analysis", self._task_analysis_node)
        for node in os.listdir(f'experiments/{self.experts_location}/slg'):
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
        for edge in os.listdir(f'experiments/{self.experts_location}/slg'):
            if not edge.endswith(".json"):
                logger.info(f"Adding edge {edge}.")
                graph_builder.add_edge(edge, END)

        graph = graph_builder.compile()
        logger.info("Graph built.")
        logger.info(40*'-')

        return graph

    # Step 4: Execute the Graph
    def ask_slg(self, file):
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

        with open(f'answers/{self.experiment}/slg.json', 'w') as f:
            json.dump(answers_list, f, indent=4)

        return None
