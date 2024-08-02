import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

st.set_page_config(page_title="Quran Chatbot, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

openai.api_key = st.secrets.openai_key

st.title("Chat with the Kaggle Quran Dataset, powered by LlamaIndex 💬🦙")
st.info("Check out the [source dataset](https://www.kaggle.com/datasets/imrankhan197/the-quran-dataset) used to build this app", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Assalamu Alaikum. Ask me anything about the Holy Quran.",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(
    input_files=["./data/Kaggle_Quran_Dataset.md"]
)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are an expert in the Quran. Answer all queries in an academic but friendly manner. Always cite references from the Quran (with surah Enlglish name and number, followed by verse(s), like this: Reference: [surah_no,]) whenever possible.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)