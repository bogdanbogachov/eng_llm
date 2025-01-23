from huggingface_hub import InferenceClient
from huggingface_hub import login

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from huggingface_hub import notebook_login
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import pipeline

from config import CONFIG
from logging_config import logger


def finetune_slg_node(node, data, experiment_number):
    logger.info(f"Finetuning {node}.")
    model_id = "local_llama_model"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=data,  split="train")
    logger.debug(f"Dataset after loading {dataset}")
    logger.debug(f"Dataset after loading {dataset.shape}")

    # Define a function to apply the chat template
    def apply_chat_template(example):
        messages = [
            {"role": "user", "content": example['question']},
            {"role": "assistant", "content": example['answer']}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt}

    # Apply the chat template function to the dataset
    new_dataset = dataset.map(apply_chat_template)
    new_dataset = new_dataset.train_test_split(0.20)
    logger.debug(f"Dataset after splitting {new_dataset}")

    # Tokenize the data
    def tokenize_function(example):
        tokens = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=512)
        # Set padding token labels to -100 to ignore them in loss calculation
        tokens['labels'] = [
            -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
        ]
        return tokens

    # Apply tokenize_function to each row
    tokenized_dataset = new_dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(['question', 'answer', 'prompt'])

    # Define training arguments
    peft_params = LoraConfig(
        lora_alpha=16,
        # lora_dropout=config['lora_dropout'],
        r=16,
        # bias=config['bias'],
        task_type='CAUSAL_LM'
    )

    training_args = TrainingArguments(
        output_dir=f"checkpoints",
        eval_strategy="steps",  # To evaluate during training
        eval_steps=40,
        logging_steps=40,
        save_steps=150,
        per_device_train_batch_size=2,  # Adjust based on your hardware
        per_device_eval_batch_size=2,
        num_train_epochs=2,  # How many times to loop through the dataset
        fp16=False,  # Must be False for MacBooks
        report_to="none",  # Here we can use something like tensorboard to see the training metrics
        log_level="error",
        learning_rate=1e-5,  # Would avoid larger values here
        max_grad_norm=2  # Clipping the gradients is always a good idea
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_params,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer
        )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    os.makedirs(f'experiments/{experiment_number}', exist_ok=True)
    trainer.save_model(f"experiments/{experiment_number}/finetuned_{node}")

    return None

def finetune_llama():
    pass
