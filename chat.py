from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from fastapi import FastAPI


llm = ChatGroq(
    api_key="YOUR_API_KEY",
    model="llama-3.1-8b-instant"
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful model for users"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm


session = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session:
        session[session_id] = ChatMessageHistory()
    return session[session_id]


with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


app = FastAPI()


