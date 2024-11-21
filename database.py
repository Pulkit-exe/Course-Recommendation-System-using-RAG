from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer

CHROMA_PATH = "chroma"

# Initialize the local embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Wrapper function for embedding compatibility with Chroma
class LocalEmbeddingFunction:
    def embed_documents(self, texts):
        # Generate embeddings locally with SentenceTransformer
        embeddings = embedding_model.encode(texts)
        # Ensure embeddings are returned as a list of lists
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

def get_embedding_function():
    return LocalEmbeddingFunction()

def load_data(url):
    loader = WebBaseLoader([url])
    transformer = Html2TextTransformer()
    docs = transformer.transform_documents(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    return splitter.split_documents(docs)

def add_to_chroma(chunks):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    valid_chunks = [chunk for chunk in chunks if chunk]  # Filter out any invalid chunks
    db.add_documents(valid_chunks)

def main():
    with open("extracted_links.txt", 'r') as file:

        links = file.readlines()

        for url in links:

            url = url.strip()
            chunks = load_data(url)
            add_to_chroma(chunks)

        print("Script ran successfully and documents were added to Chroma.")
        
if __name__ == "__main__":
    main()
