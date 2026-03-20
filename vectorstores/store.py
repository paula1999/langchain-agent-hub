import os
import time
from langchain_chroma import Chroma

class VectorStore:
    """
    This class is a wrapper around the Chroma vector store, which allows us to easily add documents and retrieve them using a retriever.
    It also handles the persistence of the vector store and the configuration of the embeddings.
    """
    def __init__(self, embeddings, persist_directory, collection_name):
        """
        Initializes the VectorStore instance.
        Args:
            - embeddings: The embeddings to use for the vector store.
            - persist_directory: The directory where the vector store will be persisted.
            - collection_name: The name of the collection to use in the vector store.
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self):
        """
        This function sets up the Chroma vector store and handles the persistence directory.
        It checks if the persistence directory exists and creates it if it doesn't. Then it initializes the Chroma vector store with the specified embeddings, persistence directory, and collection name.
        """
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            print(f"Directorio creado en: {self.persist_directory}")

        try:
            vs = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            print(f"ChromaDB '{self.collection_name}' configurado correctamente.")
            return vs
        except Exception as e:
            print(f"Error al configurar ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents):
        """
        This function adds documents to the vector store. It handles potential exceptions, such as rate limits, and retries if necessary.
        Args:
            - documents: A list of documents to add to the vector store.
        """
        try:
            self.vectorstore.add_documents(documents)
            print(f"Se han añadido {len(documents)} documentos a '{self.collection_name}'.")
        except Exception as e:
            if "429" in str(e):
                print("Cuota excedida. Esperando 60 segundos para reintentar...")
                time.sleep(65)
                self.vectorstore.add_documents(documents)
            else:
                print(f"Error al añadir documentos: {str(e)}")

    def get_retriever(self, search_type='similarity', k=3):
        """
        This function returns a retriever for the vector store. It allows you to specify the search type and the number of results to return.
        Args:
            - search_type: The type of search to perform (default is 'similarity').
            - k: The number of results to return (default is 3).
        Returns:
            - A retriever for the vector store configured with the specified search type and number of results.
        """
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={'k': k}
        )