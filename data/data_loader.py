from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

api_key="YOUR_API_KEY"
def get_link(link:str):
    web_loader=WebBaseLoader(link)
    web_loader=web_loader.load()
    return web_loader

def text_spliter(text):
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    textsplit=text_spliter.split_documents(text)
    return textsplit

def embedding_chroma(text):
    embed=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectoredb=Chroma.from_documents(documents=text,embedding=embed,persist_directory="/vectoredb/chroma_db")
    

def retrive_answer(question):
    embed=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb=Chroma(persist_directory="/vectoredb/chroma_db",embedding_function=embed)
    retriever=vectordb.as_retriever()
    responce=retriever.invoke(question)[0].page_content
    return responce

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key
    )


def run_llm(prompt: str):
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content