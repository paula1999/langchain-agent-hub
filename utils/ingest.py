import os
import time

from vectorstores.store import VectorStore

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_files():
    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    # Vector Store
    vectorstore = VectorStore(
        embeddings=embeddings,
        persist_directory='vectorstores',
        collection_name='europe'
    )

    # PDF files
    pdf_folder = "data"

    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f'PDF folder not found: {pdf_folder}')
    
    pdfs = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    for pdf in pdfs:
        # Load PDF
        pdf_path = os.path.join(pdf_folder, pdf)
        pdf_loader = PyPDFLoader(pdf_path)

        try:
            pages = pdf_loader.load()
        except Exception as e:
            print(f'Error loading PDF: {e}')
            raise

        # Chunking process
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        pages_split = text_splitter.split_documents(pages)
        
        # Batch
        batch_size = 10
        for i in range(0, len(pages_split), batch_size):
            batch = pages_split[i : i + batch_size]
            
            # Add docs to vector store
            vectorstore.add_documents(batch)

            time.sleep(6)