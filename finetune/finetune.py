import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from logging_config import logger


def finetune(model_to_tune, adapter_name, data, experiment_number, slg=False, orchestrator=False):
    logger.info(f"Finetuning {adapter_name}.")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found! Please ensure you have a CUDA-compatible GPU.")

    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        model_to_tune,
        torch_dtype=torch.float16,
    ).to(device)

    # Print a device
    logger.info(f"Model is loaded on: {model.device}")

    # Print GPU memory usage
    logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    logger.info(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    tokenizer = AutoTokenizer.from_pretrained(model_to_tune)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=data,  split="train")
    logger.debug(f"Dataset after loading {dataset}")
    logger.debug(f"Dataset after loading {dataset.shape}")

    # Define a function to apply the chat template
    def apply_chat_template(example):
        if orchestrator:
            messages = [
                {
                    "role": "user",
                    "content":
                        f"Analyze this question and find an appropriate expert who can answer it: {example['question']}"
                },
                {"role": "assistant", "content": f"{example['title'].replace(' ', '_').replace('/', '_').lower()}" }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {"prompt": prompt}
        else:
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
        tokens = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=1024)
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
        lora_dropout=0.05,
        r=32,
        task_type='CAUSAL_LM'
    )

    if orchestrator:
        learning_rate = 2e-5
        label_smoothing_factor = 0.15
    else:
        learning_rate = 1e-3
        label_smoothing_factor = 0.01

    training_args = TrainingArguments(
        output_dir=f"checkpoints",
        eval_strategy="steps",  # To evaluate during training
        eval_steps=10,
        logging_steps=10,
        save_steps=150,
        fp16=True,
        report_to="tensorboard",
        log_level="info",
        logging_dir="logs",

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=learning_rate,
        weight_decay=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=2,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        gradient_accumulation_steps=1,
        optim='adamw_torch',
        label_smoothing_factor=label_smoothing_factor
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
    trainer.evaluate()

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
