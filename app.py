import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥",
    layout="wide"
)

# Title and description
st.title("🎥 YouTube Video Chatbot")
st.markdown("Ask questions about any YouTube video by providing its URL!")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'summary_loading' not in st.session_state:
    st.session_state.summary_loading = False

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def load_transcript(video_id):
    """Load YouTube transcript"""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id=video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript, None
    except TranscriptsDisabled:
        return None, "Captions are disabled for this video."
    except Exception as e:
        return None, f"Error loading transcript: {str(e)}"

def create_vector_store(transcript):
    """Create vector store from transcript"""
    with st.spinner("Processing transcript..."):
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.create_documents([transcript])

        # Create embeddings using Google's API
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create vector store
        vector_store = FAISS.from_documents(chunks, embedding)

        return vector_store, len(chunks)

def get_answer(vector_store, question):
    """Get answer from the chatbot"""
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Create prompt
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context: {context}

        Question: {question}

        Answer:
        """,
        input_variables=['context', 'question']
    )

    # Prepare context
    content_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({"context": content_text, "question": question})

    # Get answer from LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    answer = llm.invoke(final_prompt.text)

    return answer.content, retrieved_docs

def generate_summary(transcript):
    """Generate a summary of the video transcript"""
    prompt = PromptTemplate(
        template="""
        You are an expert at summarizing video transcripts.
        Please provide a comprehensive summary of the following transcript.

        Format your response as:

        📌 **Key Points:**
        - Point 1
        - Point 2
        - Point 3
        (etc.)

        📝 **Summary:**
        [Write a 2-3 paragraph summary here]

        🎯 **Main Takeaway:**
        [One sentence main takeaway]

        Transcript:
        {transcript}
        """,
        input_variables=['transcript']
    )

    final_prompt = prompt.invoke({"transcript": transcript})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    summary = llm.invoke(final_prompt.text)

    return summary.content

# Sidebar
with st.sidebar:
    st.header("Video Settings")

    # Video URL input
    video_url = st.text_input(
        "YouTube Video URL or ID",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    load_button = st.button("Load Video", type="primary", use_container_width=True)

    if load_button and video_url:
        video_id = extract_video_id(video_url)

        if video_id:
            st.session_state.video_id = video_id

            # Load transcript
            transcript, error = load_transcript(video_id)

            if error:
                st.error(error)
            elif transcript:
                st.session_state.transcript = transcript

                # Create vector store
                vector_store, num_chunks = create_vector_store(transcript)
                st.session_state.vector_store = vector_store

                # Clear chat history
                st.session_state.chat_history = []

                st.success(f"✅ Video loaded! ({num_chunks} chunks)")
        else:
            st.error("Invalid YouTube URL or ID")

    # Display video info
    if st.session_state.video_id:
        st.divider()
        st.subheader("Current Video")
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

        if st.button("Clear Video", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.transcript = None
            st.session_state.video_id = None
            st.session_state.summary = None
            st.rerun()

# Main content area
if st.session_state.vector_store is None:
    st.info("👈 Enter a YouTube URL in the sidebar to get started!")

    # Example videos
    st.subheader("Try these example videos:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 LangChain Tutorial"):
            st.session_state.example_url = "0FDEYEFVGPk"
            st.rerun()
    with col2:
        if st.button("🤖 AI Overview"):
            st.session_state.example_url = "aircAruvnKk"
            st.rerun()

    if 'example_url' in st.session_state:
        video_url = st.session_state.example_url
        del st.session_state.example_url

else:
    # Create tabs for different sections
    chat_tab, summary_tab = st.tabs(["💬 Chat", "📋 Summary"])

    with chat_tab:
        st.subheader("Ask Questions")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("📄 View Sources"):
                        for i, doc in enumerate(message["sources"], 1):
                            st.text(f"Source {i}:\n{doc.page_content[:200]}...")

        # Chat input
        question = st.chat_input("Ask a question about the video...")

        if question:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.write(question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = get_answer(st.session_state.vector_store, question)
                    st.write(answer)

                    with st.expander("📄 View Sources"):
                        for i, doc in enumerate(sources, 1):
                            st.text(f"Source {i}:\n{doc.page_content[:200]}...")

            # Add assistant message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

    with summary_tab:
        st.subheader("Video Summary")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Get an AI-generated summary of the video content")
        with col2:
            if st.button("🔄 Generate Summary", use_container_width=True):
                st.session_state.summary_loading = True
                st.rerun()

        st.divider()

        if st.session_state.summary_loading:
            with st.spinner("✨ Generating summary... This may take a moment"):
                try:
                    summary = generate_summary(st.session_state.transcript)
                    st.session_state.summary = summary
                    st.session_state.summary_loading = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    st.session_state.summary_loading = False

        if st.session_state.summary:
            st.markdown(st.session_state.summary)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📝 Regenerate Summary", use_container_width=True):
                    st.session_state.summary_loading = True
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear Summary", use_container_width=True):
                    st.session_state.summary = None
                    st.rerun()
        else:
            st.info("👆 Click 'Generate Summary' to create an AI summary of this video")

# Footer
st.divider()
st.caption("Built with LangChain, FAISS, and Streamlit")
