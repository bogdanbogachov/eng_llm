from transformers import AutoModelForCausalLM, AutoTokenizer
from logging_config import logger
import os


def download(model_name, save_directory):
    logger.info(f"Downloading {model_name}...")


    model_name = model_name
    save_directory = save_directory

    os.makedirs(save_directory, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=False,
        trust_remote_code=True
    )

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    logger.info(f"{model_name} downloaded.")

    return None
