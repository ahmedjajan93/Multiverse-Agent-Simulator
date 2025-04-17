import streamlit as st
import graphviz
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import torch
import os

# Ensure API key is set in your environment variables (don't hardcode it)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LLM using OpenRouter (gryphe/mythomax is free)
llm = ChatOpenAI(
    model="gryphe/mythomax-l2-13b",
    base_url="https://openrouter.ai/api/v1"
)

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# FAISS store to track timeline decisions
faiss_store = FAISS.from_texts(["timeline memory vector space initialized"], embedding_model)

# 🛠 Define Structured Tools
def finance_tool(decision: str) -> str:
    return "FinanceTool: VC funding enables fast growth but includes high expectations."

def risk_tool(decision: str) -> str:
    return "RiskEvaluator: Bootstrapping offers control and safety but limits scaling."

custom_tools = [
    StructuredTool.from_function(finance_tool, name="FinanceTool", description="Analyzes financial options."),
    StructuredTool.from_function(risk_tool, name="RiskEvaluator", description="Evaluates risk of decisions.")
]

# 🚀 Generate timelines with slightly varied prompts
def generate_timelines(base_prompt, n):
    return [
        f"{base_prompt} (In this timeline, the world operates under variation #{i+1}.)"
        for i in range(n)
    ]

# 🧠 Create agents with memory, simulate decisions
def create_agents_with_decisions(timelines):
    decisions = []
    agents = []

    for timeline in timelines:
        memory = ConversationBufferMemory()
        agent = initialize_agent(
            tools=custom_tools,
            llm=llm,
            memory=memory,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )
        decision = agent.run(timeline)
        faiss_store.add_texts([decision])
        decisions.append(decision)
        agents.append(agent)

    return decisions, agents

# 🤖 Compare decisions across timelines
def compare_decisions(decisions):
    prompt = "Here are decisions from agents in alternate timelines:\n"
    for i, d in enumerate(decisions):
        prompt += f"\n{i+1}. {d}"
    prompt += "\n\nWhich decision seems most optimal and why?"
    return llm.predict(prompt)

# 🌳 Visualize timeline divergence
def visualize_divergence(timelines, decisions):
    dot = graphviz.Digraph()
    dot.node("Start", "🌍 Base Scenario")

    for i, (t, d) in enumerate(zip(timelines, decisions), 1):
        dot.node(f"T{i}", f"Timeline {i}")
        dot.edge("Start", f"T{i}")
        dot.node(f"D{i}", d[:40] + "...")
        dot.edge(f"T{i}", f"D{i}")

    st.graphviz_chart(dot)

# 🌌 Streamlit UI
st.set_page_config(page_title="🌌 Multiverse Agent Simulator", layout="wide")
st.title("🌌 Multiverse Agent Simulator")

base_prompt = st.text_area("Enter a scenario or decision-making goal:",
    "You are a startup founder deciding between bootstrapping or raising VC funding.")

num_timelines = st.slider("Number of alternate timelines:", 2, 5, 3)

if st.button("Run Multiverse Simulation"):
    with st.spinner("Generating alternate realities..."):
        timelines = generate_timelines(base_prompt, num_timelines)

    with st.spinner("Simulating agents in each timeline..."):
        decisions, agents = create_agents_with_decisions(timelines)

    st.subheader("🧠 Agent Decisions Across Timelines")
    for i, (t, d) in enumerate(zip(timelines, decisions), 1):
        with st.expander(f"🌐 Timeline {i}"):
            st.markdown(f"**Prompt:** {t}")
            st.markdown(f"**Decision:** {d}")

    st.subheader("⚖️ Comparison of Agent Decisions")
    comparison = compare_decisions(decisions)
    st.markdown(comparison)

    st.subheader("🌳 Decision Divergence Map")
    visualize_divergence(timelines, decisions)

    st.subheader("💬 Chat with Timeline Agent")
    timeline_choice = st.selectbox("Select a timeline to interact with:", [f"Timeline {i+1}" for i in range(num_timelines)])
    user_query = st.text_input("Ask your question to the selected timeline agent:")
    if user_query:
        agent_index = int(timeline_choice.split()[-1]) - 1
        reply = agents[agent_index].run(user_query)
        st.markdown(f
