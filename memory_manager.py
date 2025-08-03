import chromadb
import google.generativeai as genai

# Initialize ChromaDB client
# This creates a persistent database in the 'chroma_db' directory
client = chromadb.PersistentClient(path="./chroma_db")

# Get or create a collection for storing user memories
memory_collection = client.get_or_create_collection(name="user_memories")

# Configure the Gemini model for summarization
# Assuming genai is already configured in main.py
llm_model = genai.GenerativeModel('gemini-1.5-flash')

def get_long_term_memory(user_id: str) -> str:
    """Retrieves long-term memories for a user from the vector store."""
    try:
        results = memory_collection.get(where={"user_id": user_id}, include=["documents"])
        memories = "\n".join(doc for doc in results['documents'])
        return f"--- Past Memories ---\n{memories}\n--- End of Memories ---" if memories else ""
    except Exception as e:
        print(f"Error retrieving memory: {e}")
        return ""

def summarize_and_store_memory(user_id: str, chat_history: list):
    """Summarizes the conversation and stores key facts in the vector store."""
    if len(chat_history) < 4: # Don't summarize very short conversations
        return

    # Format the history for the summarization prompt
    formatted_history = "\n".join([f"{entry['role']}: {entry['parts'][0]}" for entry in chat_history])
    
    prompt = f"""
    Based on the following conversation, extract key personal facts about the user 
    (e.g., name, location, specific interests, important life events).
    Present these facts as a concise list. If no new key facts are revealed, respond with 'NONE'.

    Conversation:
    {formatted_history}
    """

    response = llm_model.generate_content(prompt)
    summary = response.text.strip()

    if summary != 'NONE' and summary:
        facts = [fact.strip() for fact in summary.split('\n') if fact.strip()]
        for fact in facts:
            # Use a unique ID for each fact to avoid duplicates
            fact_id = f"{user_id}_{hash(fact)}"
            memory_collection.add(
                ids=[fact_id],
                documents=[fact],
                metadatas=[{"user_id": user_id}]
            )
        print(f"Stored memories for {user_id}: {facts}")