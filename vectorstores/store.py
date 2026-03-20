import os
import time
from langchain_chroma import Chroma

class VectorStore:
    def __init__(self, embeddings, persist_directory, collection_name):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self):
        """Inicializa la base de datos Chroma, creando el directorio si no existe."""
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
        Recibe una lista de objetos Document de LangChain, 
        los añade a la colección y persiste los cambios.
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
        """Genera un retriever basado en la configuración deseada."""
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={'k': k}
        )