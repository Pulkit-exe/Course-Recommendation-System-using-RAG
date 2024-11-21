# Course Recommendation System

This project is a **Retrieval-Augmented Generation (RAG)** based recommendation system for free online courses, hosted on Hugging Face Spaces.

## Features
- **Web Scraping:** Scrapes data of 71 free courses from Analytics Vidhya.
- **Embedding Generation:** Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2` and stores them in a **Chroma database**.
- **User Interface:** Interactive Gradio chat interface for personalized course recommendations.
- **API Integration:** Utilizes the Qwen2.5-Coder-32B-Instruct model via Hugging Face serverless APIs.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   python gradio_app.py
   ```
## Files
- `chroma/` - Contains the Chroma database and related binary files.
- `database.py` - Script for managing data embeddings.
- `gradio_app.py` - Gradio chat interface for course recommendations.
- `extracted_links.txt` - Contains scraped course URLs.
- `links.py` - Helper for link extraction and processing.

## Technologies Used
- Python, BeautifulSoup, Gradio, Hugging Face, Chroma DB, SentenceTransformers

## License
This project is licensed under the MIT License.

