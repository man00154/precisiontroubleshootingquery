# app.py
import os
import asyncio
import streamlit as st
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- Load API Key from Streamlit secrets ---
# NOTE: The API key should be named GEMINI_API_KEY in your Streamlit secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Please set GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

# --- Ensure asyncio loop exists (Python 3.13 stricter rules) ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- App UI ---
st.title("MANISH - DATA CENTER Precision Troubleshooting Assistant")

uploaded_file = st.file_uploader("Upload historical incident reports (txt/csv)", type=["txt", "csv"])
user_query = st.text_input("Describe your problem:")

if uploaded_file and user_query:
    # Read file content
    content = uploaded_file.read().decode("utf-8")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(content)

    # Convert chunks to Document objects
    docs = [Document(page_content=t) for t in texts]

    # Initialize embeddings with the Gemini API key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # Create FAISS vectorstore from the documents and embeddings
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        st.stop()

    # Define a retrieval function that will be used by the agent as a tool
    def retrieve_relevant(query):
        """
        Retrieves the most relevant historical incident reports based on a user query.
        """
        results = vectorstore.similarity_search(query, k=3)
        return " ".join([r.page_content for r in results])

    # Initialize the Gemini LLM for the agent
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    # Define the agent tool
    agent_tools = [
        Tool(
            name="TroubleshootTool",
            func=retrieve_relevant,
            description="A tool that retrieves relevant historical incident reports for a given query."
        )
    ]

    # Initialize the agent
    agent = initialize_agent(
        agent_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Run agent after the user clicks the button
    if st.button("Get Resolution Plan"):
        with st.spinner("Generating expert troubleshooting plan..."):
            try:
                # The agent will now use the retrieve_relevant tool to get context
                # and then use the Gemini LLM to formulate the final answer.
                result = agent.run(
                    f"""
                    You are a virtual expert for data center troubleshooting.
                    Using the historical context, provide a step-by-step resolution plan for this user query:
                    {user_query}
                    """
                )
                st.subheader("Step-by-Step Resolution Plan")
                st.write(result)
            except Exception as e:
                st.error(f"Error running agent: {e}")
