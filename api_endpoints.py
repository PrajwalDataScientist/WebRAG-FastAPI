from fastapi import FastAPI
from pydantic import BaseModel
from data.data_loader import get_link, text_spliter, embedding_chroma, retrive_answer, run_llm

app = FastAPI()



class URLRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "Web RAG API running"}



@app.post("/load-url")
def load_url(req: URLRequest):
    # Load website
    docs = get_link(req.url)
    chunks = text_spliter(docs)
    embedding_chroma(chunks)

    return {
        "status": "success",
        "message": "Website loaded into vector database"
    }




@app.post("/ask")
def ask_question(req: QuestionRequest):
   
    context = retrive_answer(req.question)

    
    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {req.question}
    """

    answer = run_llm(prompt)

    return {
        "question": req.question,
        "answer": answer
    }
