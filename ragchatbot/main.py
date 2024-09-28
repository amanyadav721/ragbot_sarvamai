import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from models import AskQuestionRequest
from utils import rag, create_agent, get_vectorstore, sarvamai_feature


logging.basicConfig(level=logging.INFO)


load_dotenv()

app = FastAPI()

def is_math_query(question: str) -> bool:
    math_keywords = ["calculate", "sum", "difference", "multiply", "divide", "math", "what is", "how much"]
    return any(keyword in question.lower() for keyword in math_keywords)

try:
    conversation = rag()
    logging.info("Conversation chain initialized successfully.")
except Exception as e:
    conversation = None
    logging.error(f"Failed to initialize conversation chain: {e}")




try:
    agent = create_agent()
    logging.info("Agent initialized successfully.")
except Exception as e:
    agent = None
    logging.error(f"Failed to initialize agent: {e}")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    index_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
    logging.info(f"Looking for index.html at: {index_path}")
    if os.path.exists(index_path):
        logging.info("index.html found. Serving the file.")
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        logging.error("index.html not found.")
        return JSONResponse(content={"message": "index.html not found"}, status_code=404)

@app.post("/ask_question/")
async def ask_question(request: AskQuestionRequest):
    global conversation
    if not conversation:
        logging.error("Conversation chain not initialized.")
        raise HTTPException(status_code=400, detail="Conversation chain not initialized")
    
    try:
        response_ai = conversation.invoke({'question': request.question})
        
        # Log response_ai to check its structure
        logging.info(f"Response AI: {response_ai} (type: {type(response_ai)})")
        
        # Check if response_ai is a dictionary
        if isinstance(response_ai, dict):
            ai_message = response_ai.get('answer')
            if ai_message is None:
                raise ValueError("No AI response found in the response.")
            
            response = sarvamai_feature(ai_message, request.language)
            chat_history = response_ai.get('chat_history', [])
            print("iT'S AI")
            print(ai_message)
            
            if chat_history and isinstance(chat_history, list):
                logging.info(f"User asked: {request.question}")
                logging.info(f"AI responded: {chat_history[-1].content}")
                
                return JSONResponse(content={"answer": response}, status_code=200)
            else:
                raise ValueError("Chat history is not in the expected format.")
    except Exception as e:
        logging.error(f"Error during conversation invocation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/agent_action/")
async def agent_action(request: AskQuestionRequest):
    global conversation
    if not conversation:
        logging.error("Conversation chain not initialized.")
        raise HTTPException(status_code=400, detail="Conversation chain not initialized")
    
    try:
        logging.info(f"Received question: {request.question}")

        if is_math_query(request.question):
            logging.info("Identified as a math query.")
            response = conversation.invoke({'question': request.question})
        else:
            logging.info("Processing as a normal conversation query.")
            response = conversation.invoke({'question': request.question})
        
        chat_history = response['chat_history']
        logging.info(f"User asked: {request.question}")
        logging.info(f"AI responded: {chat_history[-1].content}")
        return JSONResponse(content={"answer": chat_history[-1].content}, status_code=200)
        
    except Exception as e:
        logging.error(f"Error during agent action: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8000)
