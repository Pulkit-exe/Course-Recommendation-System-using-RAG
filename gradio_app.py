from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import AIMessage, HumanMessage
from langchain_chroma import Chroma
import gradio as gr

# Load environment variables
CHROMA_PATH = "chroma"

# Hugging Face API setup
repo_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

PROMPT_TEMPLATE = """
Answer the question based on the context provided. If no relevant information is found, state so.

Context:
{context}

Question:
{question}

Answer:
"""

# Initialize the local embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


class LocalEmbeddingFunction:
    def embed_documents(self, texts):
        # Generate embeddings for a list of texts
        embeddings = embedding_model.encode(texts)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def embed_query(self, query):
        # Generate an embedding for a single query string
        query_embedding = embedding_model.encode(query)
        return query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding


class LLM:
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.2,
    )

    def generate_response(self, prompt):
        return self.llm.invoke(prompt)


def get_embedding_function():
    return LocalEmbeddingFunction()

def get_chat_response(query, history):
    # Retrieve embeddings from the Chroma DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Create prompt with context and query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # Generate response using the LLM class and Hugging Face model
    model = LLM()
    response_text = model.generate_response(prompt)

    # Update the history
    history.append(AIMessage(content = response_text))

    return response_text


# Gradio Interface
def predict(message, history):
    # Initialize history if not provided
    history_langchain_format = []

    for msg in history:

        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))

        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))

    history_langchain_format.append(HumanMessage(content=message))

    # Get response from the model
    response = get_chat_response(message, history_langchain_format)

    return response

gr.ChatInterface(predict, type="messages").launch()
