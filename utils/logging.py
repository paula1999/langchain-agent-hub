import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langchain-agent-hub")

def log(message: str):
    logger.info(message)
