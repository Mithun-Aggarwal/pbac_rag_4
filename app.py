# --- START OF FILE app.py ---

# app.py - DEPLOYMENT READY

# --- Hot-patch for sqlite3 (safe to keep for now) ---
import sys
import sqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End of hot-patch ---
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import yaml
import os
import time
import re # Import the regular expression module
from neo4j import GraphDatabase, exceptions
from pinecone import Pinecone

# Import your existing chatbot functions and utilities
from utils.logger import setup_logger
from smart_chatbot.embedder import embed_query
from smart_chatbot.generator import generate_response

# --- 1. Utility and Initialization Functions ---

@st.cache_resource
def load_backend_config():
    with open("config.yaml", 'r') as file:
        return yaml.safe_load(file)

@st.cache_resource
def load_ui_config():
    with open("ui_config.yaml", 'r') as file:
        return yaml.safe_load(file)

@st.cache_resource
def load_pinecone_index(config: dict):
    api_key = os.getenv("PINECONE_API_KEY")
    host = os.getenv("PINECONE_HOST")
    if not api_key or not host:
        st.error("PINECONE credentials not found in environment variables.")
        return None
    try:
        pc = Pinecone(api_key=api_key)
        index_name = config['vector_db'].get('collection_name', 'pbac-documents')
        index = pc.Index(name=index_name, host=host)
        print(f"Pinecone connected successfully to index: {index_name}")
        return index
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {e}")
        return None

@st.cache_resource
def get_neo4j_driver():
    try:
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        print("Connection to Neo4j AuraDB verified successfully for app.")
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

@st.cache_data(ttl=3600)
def load_graph_entities(_driver, entity_labels=['Drug', 'Sponsor', 'Condition']):
    if not _driver: return {}
    entities_by_label = {}
    with _driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
        for label in entity_labels:
            result = session.run(f"MATCH (e:{label}) RETURN e.name AS name")
            entities_by_label[label] = {record["name"].lower() for record in result if record["name"]}
    print(f"Loaded {sum(len(v) for v in entities_by_label.values())} entities from Neo4j.")
    return entities_by_label

# --- 2. ADVANCED RAG FUNCTIONS ---

def extract_entities_from_prompt(prompt: str, entities_by_label: dict) -> list[str]:
    """Finds all known entities in the prompt using whole-word matching."""
    prompt_lower = prompt.lower()
    found_entities = set()
    all_known_names = set.union(*entities_by_label.values()) if entities_by_label else set()
    
    for name in sorted(list(all_known_names), key=len, reverse=True):
        if re.search(r'\b' + re.escape(name) + r'\b', prompt_lower):
            found_entities.add(name)
            
    return list(found_entities)

def get_context_for_entity(entity_name: str, _driver, index, config: dict) -> dict:
    context = {"graph": "", "vector": "", "sources": []}
    
    cypher_query = """
    MATCH (n) WHERE toLower(n.name) = $name
    MATCH (n)-[r]-(t)
    RETURN head(labels(n)) as n_label, n.name as n_name, type(r) as rel, head(labels(t)) as t_label, t.name as t_name LIMIT 5
    """
    try:
        with _driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            result = session.run(cypher_query, name=entity_name)
            graph_parts = [f"({record['n_name']}:{record['n_label']}) -[:{record['rel']}]-> ({record['t_name']}:{record['t_label']})" for record in result]
            if graph_parts:
                graph_text = "Relationships from Knowledge Graph:\n- " + "\n- ".join(graph_parts)
                context["graph"] = graph_text
                context["sources"].append({"type": "graph", "text": graph_text})
    except Exception as e:
        print(f"Error querying graph context for {entity_name}: {e}")

    query_embedding = embed_query(f"Detailed information about {entity_name}", config)
    try:
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        vector_parts = []
        for match in results.get('matches', []):
            meta = match.get('metadata', {})
            text = meta.get('text', '')
            vector_parts.append(text)
            context["sources"].append({
                "type": "vector", "doc_title": meta.get('doc_title', 'N/A'),
                "text": text, "distance": 1 - match.get('score', 0.0)
            })
        if vector_parts:
            context["vector"] = "Relevant text from documents:\n" + "\n".join(vector_parts)
    except Exception as e:
        print(f"Error querying vector context for {entity_name}: {e}")
        
    return context

# --- 3. Load Configurations and Backend ---

config = load_backend_config()
ui_config = load_ui_config()
pinecone_index = load_pinecone_index(config)
neo4j_driver = get_neo4j_driver()
graph_entities = load_graph_entities(neo4j_driver) if neo4j_driver else {}

# --- 4. Page and Sidebar Configuration ---
st.set_page_config(page_title="AI Document Intelligence", page_icon="🧠", layout="wide")
with st.sidebar:
    st.header("About")
    st.markdown(ui_config.get('about_text', "This chatbot finds answers in your documents..."))
    st.header("System Status")
    try:
        vector_count = pinecone_index.describe_index_stats()['total_vector_count'] if pinecone_index else 0
    except Exception: vector_count = 'N/A'
    st.markdown(f"**Vector Store (Pinecone):** `{vector_count}` chunks")
    st.markdown(f"**Knowledge Graph (Neo4j):** `{sum(len(v) for v in graph_entities.values())}` entities")
    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

# --- 5. Chat Interface ---
st.title("🧠 AI Document Intelligence")
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        st.markdown(ui_config.get('welcome_message', "Hello! How can I help you?"))
    example_questions = ui_config.get('example_questions', [])
    if example_questions:
        cols = st.columns(len(example_questions))
        for i, question in enumerate(example_questions):
            with cols[i]:
                if st.button(question['title'], key=f"example_{i}"):
                    st.session_state.prompt_from_button = question['query']
                    st.rerun()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("View Retrieved Sources"):
                for source in message["sources"]:
                    with st.container(border=True):
                        if source.get("entity"): st.markdown(f"**Context for:** `{source['entity'].capitalize()}`")
                        if source.get("type") == "graph":
                            st.markdown(f"**Source Type:** `Knowledge Graph`"); st.info(source.get("text"))
                        else:
                            st.markdown(f"**Source Document:** `{source.get('doc_title', 'N/A')}`")
                            st.markdown(f"**Relevance Score (Similarity):** `{1 - source.get('distance', 1.0):.4f}`")
                            st.caption(f"Retrieved Text Snippet:\n> {source.get('text', '')}")

# --- 6. Handle User Input and Generate Response ---
if "prompt_from_button" in st.session_state:
    prompt = st.session_state.prompt_from_button
    del st.session_state.prompt_from_button
else:
    prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        sources_for_display = []
        final_context_for_llm = ""
        with st.status("Thinking...", expanded=True) as status:
            status.update(label="Step 1: Identifying key entities...")
            entities = extract_entities_from_prompt(prompt, graph_entities)
            
            if not entities:
                status.update(label="No specific entities found. Performing broad search...")
                query_embedding = embed_query(prompt, config)
                context_chunks = retrieve_relevant_chunks(query_embedding, pinecone_index, top_k=5)
                final_context_for_llm = "\n\n".join(context_chunks.get("documents", [[]])[0])
                # We need to populate sources_for_display here as well for consistency
            
            elif len(entities) == 1:
                entity = entities[0]
                status.update(label=f"Step 2: Retrieving context for '{entity.capitalize()}'...")
                context_data = get_context_for_entity(entity, neo4j_driver, pinecone_index, config)
                final_context_for_llm = f"{context_data['graph']}\n\n{context_data['vector']}"
                sources_for_display.extend(context_data['sources'])
            else:
                status.update(label=f"Step 2: Analyzing multiple entities: {', '.join(e.capitalize() for e in entities)}...")
                all_contexts = []
                for entity in entities:
                    status.update(label=f"Retrieving context for '{entity.capitalize()}'...")
                    context_data = get_context_for_entity(entity, neo4j_driver, pinecone_index, config)
                    for source in context_data['sources']: source['entity'] = entity
                    sources_for_display.extend(context_data['sources'])
                    context_block = f"--- CONTEXT FOR {entity.upper()} ---\n{context_data['graph']}\n\n{context_data['vector']}\n"
                    all_contexts.append(context_block)
                final_context_for_llm = "\n".join(all_contexts)
            
            status.update(label="Step 3: Synthesizing final answer...")
            
            context_for_generator = {"documents": [[final_context_for_llm]]}
            
            response_text = generate_response(
                prompt, context_for_generator, config, chat_history=st.session_state.messages
            )
            status.update(label="Response generated!", state="complete", expanded=False)

        st.markdown(response_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources_for_display
    })
    
    st.rerun()

# --- END OF FILE app.py ---