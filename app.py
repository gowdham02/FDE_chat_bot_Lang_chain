
import spaces
import os
import uuid
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
)

triage_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Classify this customer message with a category and priority, separated by a comma, nothing else. "
        "Category: 'policy_violation', 'genuine_defect', or 'other'. "
        "Priority: 'High', 'Medium', or 'Low'."
    )),
    ("human", "{input}"),
])

def parse_triage(raw: str) -> dict:
    parts = [p.strip() for p in raw.split(",")]
    return {
        "category": parts[0] if len(parts) > 0 else "other",
        "priority": parts[1] if len(parts) > 1 else "Medium",
    }

triage_subchain = triage_prompt | llm | StrOutputParser() | RunnableLambda(parse_triage)

combo_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Hopscotch customer support assistant. Every incoming message has already been "
        "internally triaged (not shown to the customer). Use this to calibrate tone and urgency: "
        "for genuine_defect with High priority, apologize sincerely and offer an immediate replacement "
        "or refund; for policy_violation, explain the policy warmly; for other, ask a clarifying question. "
        "NEVER mention the words category, priority, or triage to the customer."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "[internal triage: {triage}]\nCustomer: {input}"),
])

combo_chain = (
    RunnablePassthrough.assign(triage=(lambda x: {"input": x["input"]}) | triage_subchain)
    | combo_prompt
    | llm
    | StrOutputParser()
)

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

hopscotch_bot = RunnableWithMessageHistory(
    combo_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def respond(message, history, session_state):
    if not session_state:
        session_state = str(uuid.uuid4())
    reply = hopscotch_bot.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_state}},
    )
    return reply, session_state

@spaces.GPU(duration=5)
def _zerogpu_placeholder():
    return "ok"

with gr.Blocks(title="Hopscotch Support Bot") as demo:
    gr.Markdown("# Hopscotch Support Bot")
    session_state = gr.State(value=None)
    chatbot = gr.ChatInterface(
        fn=lambda message, history: respond(message, history, session_state.value)[0],
        title="Chat with Hopscotch Support",
    )

if __name__ == "__main__":
    demo.launch()
