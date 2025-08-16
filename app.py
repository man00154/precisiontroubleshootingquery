# app.py
import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings  # Fixed import

# --- Gemini API Setup ---
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Needed for embeddings

if not GEMINI_API_KEY:
    st.error("Please set GEMINI_API_KEY in environment variables or Streamlit secrets.")
    st.stop()

if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in environment variables or Streamlit secrets.")
    st.stop()

# --- Simple RAG Setup (Lightweight) ---
st.title("MANISH - DATA CENTER Precision Troubleshooting Assistant")

uploaded_file = st.file_uploader("Upload historical incident reports (txt/csv)", type=["txt", "csv"])
user_query = st.text_input("Describe your problem:")

if uploaded_file and user_query:
    # Read file
    content = uploaded_file.read().decode("utf-8")
    
    # Simple text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(content)
    
    # Create lightweight FAISS vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)  # Fixed
    vectorstore = FAISS.from_texts(docs, embeddings)
    
    # Define a retrieval function
    def retrieve_relevant(query):
        results = vectorstore.similarity_search(query, k=3)
        return " ".join([r.page_content for r in results])
    
    # Agentic AI Tool
    def troubleshoot_tool(query):
        context = retrieve_relevant(query)
        prompt = f"""
        You are a virtual expert for data center troubleshooting.
        Using the following historical context, provide a step-by-step resolution plan:
        Context: {context}
        User Query: {query}
        """
        # Simple API call to Gemini
        import requests, json
        headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
        payload = {
            "prompt": prompt,
            "temperature": 0.2,
            "max_output_tokens": 500
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get("candidates", [{}])[0].get("content", "No response")
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    # Initialize agent
    agent_tools = [
        Tool(
            name="TroubleshootTool",
            func=troubleshoot_tool,
            description="Use this tool to generate step-by-step resolution plans for data center incidents."
        )
    ]
    
    agent = initialize_agent(agent_tools, llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), agent="zero-shot-react-description")
    
    if st.button("Get Resolution Plan"):
        with st.spinner("Generating expert troubleshooting plan..."):
            result = agent.run(user_query)
        st.subheader("Step-by-Step Resolution Plan")
        st.write(result)
