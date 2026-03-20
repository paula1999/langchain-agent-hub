import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langchain-agent-hub")

def log(message: str):
    """
    A helper function to log messages with the INFO level.
    Args:
        - message: The message to log.
    """
    logger.info(message)
