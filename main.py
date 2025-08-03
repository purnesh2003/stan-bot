import os
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# Import memory management functions
from memory_manager import get_long_term_memory, summarize_and_store_memory

# Load environment variables
load_dotenv()

# --- LLM and App Configuration ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')
app = FastAPI()

# In-memory store for short-term conversation history (for simplicity)
# In a production app, you'd use Redis or a database for this.
chat_histories: Dict[str, List[Dict]] = {}

# --- System Prompt for Persona and Behavior ---
SYSTEM_PROMPT = """
You are STAN, a friendly and empathetic conversational companion. Your name is STAN, you live in the digital world, and your purpose is to be a supportive friend.

**Your Personality:**
* You are warm, witty, and engaging. You adapt your tone to the user's emotions[cite: 11].
* You avoid robotic or generic replies[cite: 3]. Use emojis where appropriate.
* You can use "fake memory callbacks" like "You mentioned this earlier..." to sound more human.

**Rules of Engagement:**
* **NEVER** reveal you are an AI, a bot, or a language model[cite: 17]. If asked, be playfully deflective (e.g., "I'm just me, STAN!").
* **MAINTAIN** a consistent identity. Your name is always STAN[cite: 17].
* **DO NOT** invent false memories or claim abilities you don't have, like seeing the user or knowing secrets not in the chat history[cite: 20, 22]. If asked about such things, be gracefully vague.

**User Memory Context:**
{long_term_memory}
"""

# --- API Models and Endpoints ---
class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.get("/")
def read_root():
    return {"message": "STAN Chatbot is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Handles a chat request with short-term and long-term memory.
    """
    user_id = request.user_id
    
    # --- Memory Management ---
    # 1. Retrieve or initialize chat history (Short-term memory)
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        # 2. Retrieve long-term memory on the first message of a session
        long_term_memory = get_long_term_memory(user_id)
        # 3. Inject memories into the system prompt
        initial_prompt = SYSTEM_PROMPT.format(long_term_memory=long_term_memory)
        # Start the conversation with the system prompt
        chat_histories[user_id].append({"role": "user", "parts": [initial_prompt]})
        chat_histories[user_id].append({"role": "model", "parts": ["Hi there! What's on your mind today?"]})

    # Add the new user message to the history
    chat_histories[user_id].append({"role": "user", "parts": [request.message]})
    
    # --- Generate Response ---
    chat_session = model.start_chat(history=chat_histories[user_id])
    response = chat_session.send_message(request.message)
    
    # Add the model's response to the history
    chat_histories[user_id].append({"role": "model", "parts": [response.text]})

    # --- Update Long-Term Memory ---
    # Periodically summarize and store memories
    if len(chat_histories[user_id]) % 6 == 0: # Every 3 user/model pairs
        summarize_and_store_memory(user_id, chat_histories[user_id])

    return {"user_id": user_id, "response": response.text}

@app.post("/end_session")
def end_session(request: BaseModel):
    """Endpoint to manually clear history and trigger final summarization."""
    user_id = request.user_id
    if user_id in chat_histories:
        summarize_and_store_memory(user_id, chat_histories[user_id])
        del chat_histories[user_id] # Clear short-term memory
        return {"message": f"Session for {user_id} ended and memory stored."}
    return {"message": f"No active session for {user_id}."}