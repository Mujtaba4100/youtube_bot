# YouTube Chatbot - Streamlit UI

A Streamlit-based chatbot that lets you ask questions about any YouTube video using RAG (Retrieval Augmented Generation).

## Features

- 🎥 Load any YouTube video by URL or ID
- 💬 Ask questions about the video content
- 📄 View source references from the transcript
- 🎯 Powered by Google Gemini and LangChain

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API key:**
   - Make sure `.env` file exists with your Google API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter a YouTube URL in the sidebar (or just the video ID)
2. Click "Load Video" to process the transcript
3. Ask questions in the chat interface
4. View AI-generated answers with source citations

## How it Works

1. **Transcript Extraction**: Fetches video transcript using YouTube Transcript API
2. **Text Chunking**: Splits transcript into manageable chunks
3. **Embeddings**: Creates vector embeddings using Google's embedding model
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search
5. **Retrieval**: Finds relevant chunks based on your question
6. **Generation**: Uses Google Gemini to generate contextual answers

## Technologies

- **Streamlit**: Web UI framework
- **LangChain**: LLM orchestration
- **FAISS**: Vector similarity search
- **Google Gemini**: Language model & embeddings


## Notes

- Videos must have English captions/transcripts enabled
- First load may take longer as the embedding model downloads
- Chat history is maintained per video session

## 🌐 Deployment

Ready to share your chatbot? Deploy it on Hugging Face Spaces for free!

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete instructions on deploying to:
- Hugging Face Spaces (easiest)
- GitHub integration (recommended)
- Other Streamlit hosting options

