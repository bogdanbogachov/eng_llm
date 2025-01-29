import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from logging_config import logger


def finetune(model_to_tune, adapter_name, data, experiment_number, slg=True):
    logger.info(f"Finetuning {adapter_name}.")
    model = AutoModelForCausalLM.from_pretrained(
        model_to_tune,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_to_tune)
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
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        fp16=False,
        report_to="tensorboard",
        log_level="info",
        logging_dir="logs",
        learning_rate=1e-5,
        max_grad_norm=2
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
    if slg:
        os.makedirs(f'experiments/{experiment_number}/slg', exist_ok=True)
        trainer.save_model(f"experiments/{experiment_number}/slg/finetuned_{adapter_name}")
        with open(f"experiments/{experiment_number}/slg/finetuned_{adapter_name}/training_log.txt", "a") as log_file:
            log_file.write(str(trainer.state.log_history))
    else:
        os.makedirs(f'experiments/{experiment_number}', exist_ok=True)
        trainer.save_model(f"experiments/{experiment_number}/finetuned_{adapter_name}")
        with open(f"experiments/{experiment_number}/finetuned_{adapter_name}/training_log.txt", "a") as log_file:
            log_file.write(str(trainer.state.log_history))

    return None
