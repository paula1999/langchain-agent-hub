from vectorstores.store import VectorStore

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Embeddings and vectorstore setup
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT"
)

vectorstore = VectorStore(
    embeddings=embeddings, 
    persist_directory='../vectorstores',
    collection_name='europe'
)


@tool("retriever_tool", description="A tool to retrieve relevant information from the database")
def retriever_tool(query: str) -> str:
    """
    This tool uses the vectorstore to retrieve relevant information from the database based on the query.
    Args:
        - query: The query to retrieve relevant information for.
    Returns:
        - A string with the retrieved information or an error message if no relevant information is found.
    """
    retriever = vectorstore.get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return 'I found no relevant information in the documents.'
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f'Document {i+1}:\n{doc.page_content}')
    
    return '\n\n'.join(results)
