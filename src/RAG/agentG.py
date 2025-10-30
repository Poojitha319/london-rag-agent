# src/RAG/langgraph_agent.py
import os
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import Graph
from src.RAG.rag_engine import RAGStore
from src.RAG.utils import load_processed_df
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key="your-api-key")

# Initialize RAG store
rag = RAGStore()

# -----------------------------
# Helper: Gemini LLM wrapper
# -----------------------------
def gemini_response(prompt: str, model_name="gemini-2.0-flash"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip() if response and response.text else "No response."

# -----------------------------
# Node Functions
# -----------------------------
def clarify(state: dict):
    """Clarify ambiguous user query"""
    question = state["question"]
    msg = f"Clarify this real estate query: '{question}'. Make it specific and unambiguous."
    clarified = gemini_response(msg)
    print("🧩 Clarified Query:", clarified)
    return {"clarified": clarified}

def plan(state: dict):
    """Plan which tool or filter to use"""
    clarified = state["clarified"]
    msg = (
        f"Analyze this clarified query: '{clarified}'. "
        "Decide what to do next: retrieve properties, filter by borough, or compute stats. "
        "Respond in JSON format like: {'tool': 'retrieve', 'filters': {'borough': 'Camden', 'max_price': 500000}}"
    )
    plan_text = gemini_response(msg)
    print("🧠 Plan:", plan_text)
    return {"plan": plan_text}

def execute(state: dict):
    """Run retrieval using RAG pipeline"""
    if rag.df is None:
        df = load_processed_df()
        rag.build_index(df)

    clarified = state["clarified"]
    results = rag.query(clarified, k=5)
    print(f"🔍 Retrieved {len(results)} results")
    return {"retrieved": results}

def respond(state: dict):
    """Generate final natural answer"""
    retrieved = state["retrieved"]
    context = "\n".join([r["snippet"] for r in retrieved])
    question = state["question"]

    msg = f"""
    You are a helpful London real estate expert.
    Context:
    {context}

    User question: {question}

    Based on the above context, provide a detailed, factual answer including property IDs and prices.
    """
    answer = gemini_response(msg, model_name="gemini-1.5-pro")
    print("💬 Final Answer:", answer)
    return {"answer": answer}

# -----------------------------
# Graph Definition
# -----------------------------
def build_agent_graph():
    graph = Graph()
    graph.add_node("clarify", clarify)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("respond", respond)

    graph.add_edge("clarify", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "respond")

    graph.set_entry_point("clarify")
    graph.set_finish_point("respond")
    return graph

# -----------------------------
# Agent Runner
# -----------------------------
def run_agent(question: str):
    graph = build_agent_graph()
    state = {"question": question}
    final_state = graph.run(state)
    return final_state
