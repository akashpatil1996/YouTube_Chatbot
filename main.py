from youtube_transcript_api import YouTubeTranscriptApi
from punctuators.models import PunctCapSegModelONNX
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
import re
import os
import time
import requests


os.environ["OPENAI_API_KEY"] = "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"

def get_video_id(url):
    return re.findall(r'watch\?v=([\w-]+)', url)[0]

# Define function to retrieve transcript and punctuate it
def get_transcript(url):
    video_id = get_video_id(url)
    t = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ''
    for i in t:
        transcript += ' '+i['text']
    final_transcript = punctuate_text(transcript)
    final_transcript = "\n".join(final_transcript)
    return final_transcript

@st.cache_resource
def load_punctuate_model():
    return PunctCapSegModelONNX.from_pretrained("pcs_47lang")

def punctuate_text(text):
    m = load_punctuate_model()
    input_texts: list[str] = [text]
    results: list[list[str]] = m.infer(input_texts)
    return results[0]


# Define Streamlit app
# st.title("▶️ YouTube Chatbot 🤖",)
st.markdown("<h1 style='text-align: center;'>▶️ YouTube Chatbot 🤖</h1>", unsafe_allow_html=True)

# Create input box for YouTube URL
url = st.text_input("Enter YouTube URL", placeholder="Paste here")

# Create button to get transcript and punctuate it
if st.button("Load"):
    try:
        transcript = get_transcript(url)
        progress_text = "Loading..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Get the YouTube video information
        api_key = "AIzaSyAV_0VQNlDz_1Anbvl5nhEVlkPz4r3I310"
        video_id = get_video_id(url)
        video_info = requests.get(f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=snippet&key={api_key}").json()["items"][0]["snippet"]
        video_title = video_info["title"]
        video_thumbnail = video_info["thumbnails"]["high"]["url"]

        # Show the video information
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(video_thumbnail, width=150)
        st.write(video_title)

        # Remove progress bar
        my_bar.empty()

    except:
        st.warning('Please enter a valid URL')


# Create input box for user's question
question = st.text_input("Ask your question", placeholder="Question")

# Create button to ask question and retrieve answer
if st.button("Ask"):
    # try:
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
        transcript = get_transcript(url)
        texts = text_splitter.split_text(transcript)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_texts(texts, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="map_reduce", retriever=retriever, return_source_documents=True)
        result = qa({"query": question})

        progress_text = "Processing..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.002)
            my_bar.progress(percent_complete + 1, text=progress_text)
            # Remove progress bar
        my_bar.empty()
        st.success(result['result'])
    # except:
        # st.error('There was some error')