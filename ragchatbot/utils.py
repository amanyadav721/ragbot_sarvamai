import os
import logging
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,LLMMathChain
from langchain.agents import Tool, AgentExecutor, AgentType
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_token = os.getenv("GEMINI_KEY")
sarvamai_api_key = os.getenv("SARVAMAI_API_KEY")

# Check if the API keys are set
if not groq_api_key or not google_api_token and sarvamai_api_key:
    logging.error("GROQ_API_KEY, GEMINI_KEY,sarvamai_api_key must be set in the .env file")
    raise ValueError("GROQ_API_KEY, GEMINI_KEY,must be set in the .env file")

# Set the GROQ_API_KEY environment variable
os.environ["GROQ_API_KEY"] = groq_api_key

def sarvamai_feature(text, target_language):
    url = "https://api.sarvam.ai/translate"

    payload = {
        "input": text,
        "source_language_code": "en-IN",
        "target_language_code": target_language,
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True
    }
    headers = {"Content-Type": "application/json", 'API-Subscription-Key': sarvamai_api_key}

    response = requests.post(url, json=payload, headers=headers)
    response = response.json().get("translated_text")
    print(response)
    return response

    


# Initialize the language models
try:
    model = ChatGroq(temperature=0.6, model_name="llama-3.1-70b-versatile")
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_token)
    llm = model
    logging.info("Language models initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize language models: {e}")
    raise e

# Path to the PDF file
pdf_path = os.path.join(os.path.dirname(__file__), 'dataset', 'iesc111.pdf')

def get_pdf_text(pdf_path):
    """Extract text from a single PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        logging.info("PDF text extraction successful.")
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise e
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        logging.info(f"Text split into {len(chunks)} chunks.")
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        raise e
    return chunks

def get_vectorstore(text_chunks):
    """Create a FAISS vector store from text chunks."""
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=emb)
        logging.info("FAISS vector store created successfully.")
    except Exception as e:
        logging.error(f"Error creating FAISS vector store: {e}")
        raise e
    return vectorstore

def get_conversation_chain(vectorstore):
    """Initialize the conversational retrieval chain with memory."""
    try:
        memory =  ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        logging.info("Conversational retrieval chain initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing conversational retrieval chain: {e}")
        raise e
    return conversation_chain

def rag():
    """Load PDF, process it, and initialize the conversation chain."""
    try:
        # Load and process the PDF
        pdf_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(pdf_text)
        vectorstore = get_vectorstore(text_chunks)
        
        # Create conversation chain
        conversation_chain = get_conversation_chain(vectorstore)
        logging.info("RAG (Retrieve and Generate) pipeline initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing RAG pipeline: {e}")
        raise e
    
    return conversation_chain

def create_agent():
    """Create an Agent with VectorDB and Math Calculation tools."""
    try:
        # Initialize memory for the Agent
        pdf_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(pdf_text)
        vectorstore = get_vectorstore(text_chunks)
        
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )
        
        # Initialize the retrieval chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        # Define the VectorDB Tool with decision logic
        def vector_db_func(q):
            if any(keyword in q.lower() for keyword in ["pdf", "document", "information"]):
                return conversation_chain({"question": q})["chat_history"][-1].content
            else:
                return "No need to query the VectorDB for this input."
        
        vector_db_tool = Tool(
            name="VectorDB",
            func=vector_db_func,
            description="Useful for answering questions based on the content of the provided PDF document."
        )
        

        math_chain = LLMMathChain.from_llm(llm, verbose=False)
        math_tool = Tool(
            name="Calculator",
            func=math_chain.run,
            description="Useful for performing mathematical calculations."
        )
        
        greeting_tool = Tool(
            name="Greeting",
            func=lambda _: "Hello! How can I assist you today?",
            description="Responds with a friendly greeting."
        )
        

        tools = [vector_db_tool, math_tool, greeting_tool]
        

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            verbose=True
        )
        
        logging.info("Agent initialized successfully with VectorDB, Calculator, and Greeting tools.")
    except Exception as e:
        logging.error(f"Error creating agent: {e}")
        raise e
    
    return agent_executor

