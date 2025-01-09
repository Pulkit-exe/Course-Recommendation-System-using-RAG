import requests
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# Load environment variables
CHROMA_PATH = "chroma"

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct"
API_TOKEN = ""  # Replace with your token

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
        embeddings = embedding_model.encode(texts)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def embed_query(self, query):
        query_embedding = embedding_model.encode(query)
        return query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding

def get_embedding_function():
    return LocalEmbeddingFunction()

def get_chat_response(query):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search for relevant context
    results = db.similarity_search_with_score(query, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Create the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # Make the request to Hugging Face API
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    data = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.2,
            "max_new_tokens": 512
        }
    }
    response = requests.post(API_URL, headers=headers, json=data)

    # Handle response
    if response.status_code == 200:
        result = response.json()
        return result[0].get("generated_text", "No response generated.").replace(prompt, "").strip()
    else:
        return f"Error: {response.status_code} - {response.json()}"

def process_questions_file(input_file, output_file):
    with open(input_file, "r") as file:
        questions = file.readlines()

    answers = []
    for question in questions:
        question = question.strip()
        if question:
            response = get_chat_response(question)
            answers.append(f"Q: {question}\nA: {response}\n\n\n")

    with open(output_file, "w") as file:
        file.writelines(answers)

if __name__ == "__main__":
    input_file = "questions.txt"
    output_file = "answers.txt"
    process_questions_file(input_file, output_file)
