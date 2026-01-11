from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from data.data_loader import run_llm


SESSION_STORE = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]



prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def run_with_history(user_id: str, question: str):
    history = get_session_history(user_id)

    
    chat_history_text = "\n".join(
        [f"{m.type}: {m.content}" for m in history.messages]
    )

    answer = run_llm(chat_history_text, question)

    
    history.add_user_message(question)
    history.add_ai_message(answer)

    return answer
