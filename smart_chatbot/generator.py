# --- START OF FILE smart_chatbot/generator.py ---

import os
import google.generativeai as genai

def generate_response(prompt: str, context_chunks: dict, config: dict, chat_history: list = None):
    """
    Generates a response using the Gemini model in a conversational context,
    incorporating context from RAG and previous conversation turns.
    """
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        return "Error: GOOGLE_API_KEY not found. Please set it in your .env file."

    model_name = config.get('generation', {}).get('gemini', {}).get('model', 'gemini-1.5-flash')
    model = genai.GenerativeModel(model_name)

    # --- 1. Construct the System Prompt and RAG Context ---
    context_str = ""
    if context_chunks.get("documents") and context_chunks["documents"][0]:
        # Use the single, pre-compiled context from app.py
        context_str = context_chunks['documents'][0][0]
    else:
        context_str = "No relevant documents found for this query."
        
    system_prompt = f"""
You are an expert AI assistant specializing in analyzing Pharmaceutical Benefits Advisory Committee (PBAC) documents.
Your task is to answer the user's question based *only* on the provided context from the documents and the conversation history.
Be concise, factual, and helpful. Do not make up information.
If the context does not contain the answer, explicitly state that the information is not available in the provided documents.
"""

    # --- 2. Build the full conversational history for the API ---
    generation_input = []
    
    generation_input.append({'role': 'user', 'parts': [system_prompt]})
    generation_input.append({'role': 'model', 'parts': ["Understood. I will act as an expert AI assistant and answer based only on the provided context and history."]})

    if chat_history:
        for message in chat_history[:-1]: 
            role = 'model' if message['role'] == 'assistant' else 'user'
            generation_input.append({'role': role, 'parts': [message['content']]})
    
    # --- 3. Add the final user prompt with the RAG context ---
    final_user_prompt = f"""
CONTEXT FROM DOCUMENTS:
{context_str}
------------------
Based on the conversation history and the context above, answer this question: {prompt}
"""
    generation_input.append({'role': 'user', 'parts': [final_user_prompt]})
    
    try:
        response = model.generate_content(generation_input)
        return response.text
    except Exception as e:
        print(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."

# --- END OF FILE smart_chatbot/generator.py ---