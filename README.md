# 🦙📚 LlamaIndex - Chat with the Streamlit docs

A modified Streamlit chatbot to chat with a Quran dataset. 

## Overview of the App

- Takes user queries via Streamlit's `st.chat_input` and displays both user queries and model responses with `st.chat_message`
- Uses LlamaIndex to load and index data and create a chat engine that will retrieve context from that data to respond to each user query

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llamaindex-chat-with-docs.streamlit.app/)

## Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.
4. Add your API key to your `secrets.toml` file. If you don't already have a `secrets.toml` file, add a folder named `.streamlit`, create a file called `secrets.toml` within the folder, and add the following to it:
``` openai_key = <your key here> ```
   
Alternatively, you can use [Streamlit Community Cloud's secrets management feature](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) to add your API key via the web interface.