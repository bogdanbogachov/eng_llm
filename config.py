from pydantic import BaseModel
from logging_config import logger
import yaml
import os


class Config(BaseModel):
    @staticmethod
    def load_config(config_file: str) -> "Config":
        try:
            logger.info("Loading config file: %s", config_file)
            with open(config_file, encoding="utf-8") as f:
                parameters = yaml.safe_load(f)

                return parameters
        except OSError as exc:
            msg = f"Config file not found: {config_file}"
            raise RuntimeError(msg) from exc




CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
CONFIG = Config.load_config(CONFIG_PATH)
