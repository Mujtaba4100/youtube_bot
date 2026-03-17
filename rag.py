from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


# Step1a 
# Indexing -> using utube api we will extract transcript of video and load as string in code 
try:
    video_id="0FDEYEFVGPk" #only video id v=******* not full url 
    api=YouTubeTranscriptApi()
    transcript_list=api.fetch(video_id=video_id,languages=["en"])
    #print(transcript_list)
    transcript=" ".join(chunk.text for chunk in transcript_list)
    #print(transcript)
except:
    print("captions not found")


#Step1b
#Text spltting into chunks

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.create_documents([transcript])

print(len(chunks))

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=FAISS.from_documents(chunks,embedding)

retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":2})

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

content_text="\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt=prompt.invoke({"context":content_text,"question":question}) nmcn