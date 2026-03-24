# langchain-agent-hub

A small LangChain agent hub used to experiment with agent configurations, tools, and vector stores.

**Contents:** code, agent configs, tools, utilities, and a local Chroma vectorstore.

**Prerequisites**
- Docker (to build/run containerized image)
- Python 3.11 (for local development)
- Recommended: virtualenv or other Python env manager

**Quick start — Docker Compose**
Build the image from the repo root:

```bash
docker-compose up --build
```

Run the container:

```bash
docker-compose up
```

Stop the container:

```bash
docker-compose down
```


**Quick start — Docker**
Build the image from the repo root:

```bash
docker build -t langchain-agent-hub .
```

Run the container (pass any required runtime env vars):

```bash
docker run --rm -it \
	-e OPENAI_API_KEY="$OPENAI_API_KEY" \
	-e AEMET_API_KEY="$AEMET_API_KEY" \
	langchain-agent-hub
```

If you need to mount local data or persist the vector DB, add `-v` mounts. Example mounting `vectorstores` for persistence:

```bash
docker run --rm -it \
	-v "$PWD/vectorstores":/app/vectorstores \
	-e OPENAI_API_KEY="$OPENAI_API_KEY" \
	langchain-agent-hub
```

**Quick start — Local (dev)**
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the project:

```bash
python main.py
```
