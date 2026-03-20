from dotenv import load_dotenv

# Env vars
load_dotenv()

from agents.main_agent import create_graph, running_agent
from utils.ingest import load_files

if __name__ == '__main__':
    # Ingest files to DB
    load_files()

    rag_agent = create_graph()

    # Agent
    running_agent(rag_agent)