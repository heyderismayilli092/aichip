#-------------------------------------------------------
# File send and embedding api:
# curl -X POST http://127.0.0.1:8291/api/ragembed \
#  -F "file=@dnmsources/DECAP010_PROGRAMMING_IN_C.pdf" \
#  -F "index_name=my_rag"

#-------------------------------------------------------
# File send and delete from database api:
# curl -X POST http://127.0.0.1:8291/api/embeddelete \
#  -F "file=@dnmsources/DECAP010_PROGRAMMING_IN_C.pdf" \
#  -F "index_name=my_rag"

# -----------------------------------------
# Ask AI:
# curl -X POST http://127.0.0.1:8291/api/askai \
#  -H "Content-Type: application/json" \
#  -d '{
#    "systemprompt": "Answer the question according to the context given below. If you cannot find a suitable source to answer the question, try to answer according to your own knowledge and tell what type of sources are needed.",
#    "question": "what is machine learning ?",
#    "rag_name": "my_rag",
#    "top_k": 5,
#    "ragsource": False
#  }'

# -----------------------------------------
# Ask AI (Real-time response sending (only applicable for a suitable web interface)):
# curl -X POST http://127.0.0.1:8291/api/askai_stream \
#  -H "Content-Type: application/json" \
#  -d '{
#    "systemprompt": "Answer the question according to the context given below. If you cannot find a suitable source to answer the question, try to answer according to your own knowledge and tell what type of sources are needed.",
#    "question": "what is machine learning ?",
#    "rag_name": "my_rag",
#    "top_k": 5,
#    "ragsource": False
#  }'

# -----------------------------------------
# Get all RAG lists:
# curl -X GET http://127.0.0.1:8291/api/listrag

# -----------------------------------------
# A source embedding check with hash value:
# curl -X POST http://127.0.0.1:8291/api/checkembed \
#  -H "Content-Type: application/json" \
#  -d '{
#    "rag_name": "my_rag",
#    "hashvalue": "hash_value"
#  }'

# -----------------------------------------
# It checks if the software is up and running:
# curl -X GET http://127.0.0.1:8291/api/health

# -----------------------------------------
# Selected RAG database delete:
# curl -X POST http://127.0.0.1:8291/api/deleterag \
#  -H "Content-Type: application/json" \
#  -d '{
#    "rag_name": "my_rag"
#  }'





from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import langchain_embedding  # RAG embed with Langchain+ChromDB script
import langchain_ask  # Ask AI with Langchain + RAG
import os
import sys
import json
import time
import shutil
# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

# loading config file
with open("config.json", "r") as configfile:  # loading config file
  data = json.load(configfile)
rag_folder = data["rag-folder"]  # RAG database directory
embedding_model = data["embed-model"]  # Embedding model
chat_model = data["chat-model"]
n_ctx = data["n-ctx"]

# -------------------- LLM Settings --------------------
# Run model with CPU
#llm = LlamaCpp(
#    model_path=data["chat-model"],  # Local LLM
#    n_ctx=8192,  # Context window: with long answers and rag context
#    n_threads=6,  # CPU thread
#    n_gpu_layers=1,  # GPU use disable (optional)
#    f16_kv=True,  # Half precision key/value -- speed and RAM optimization
#    max_tokens=data["max-tokens"],  # Maximum number of tokens generated in the answer

#    # ---------- Repeat Prevention and Stable Sampling Settings ----------
#    temperature=0.4,  # Low => a more consistent and non-repetitive answer
#    top_k=30,  # Sampling top-k => controls token diversity
#    top_p=0.9,  # Sampling top-p => diversity through nuclear sampling
#    repeat_penalty=1.18,  # The most important parameter: it penalizes duplicate tokens
#    repeat_last_n=256,  # It applies repeat_penalty to the last 256 tokens generated
#    stop=["###", "Context:", "Question:"],  # Stop token -- it stops when the prompt is interrupted
#    verbose=False
#)

# --------FUNCTIONS-----------
# RAG MANAGER
class RAGManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        # Opened RAGs will be kept here
        self.cache = {}
        # Currently active RAG
        self.current_path = None
        self.current_retriever = None

    def load(self, rag_path, top_k):
        rag_path = os.path.abspath(rag_path)
        if not os.path.exists(rag_path):
            raise FileNotFoundError(f"RAG not found: {rag_path}")
        # Opened RAGs will be kept here
        if rag_path in self.cache:
            print(f"[RAG] Cache used: {rag_path}")
            self.current_path = rag_path
            self.current_retriever = self.cache[rag_path]
            return


        # Opening for the first time
        print(f"[RAG] Loading: {rag_path}")
        start = time.perf_counter()
        vectordb = Chroma(persist_directory=rag_path, embedding_function=self.embeddings)
        retriever = vectordb.as_retriever(
            search_kwargs={"k": top_k}
        )
        # Put in cache
        self.cache[rag_path] = retriever
        self.current_path = rag_path
        self.current_retriever = retriever
        elapsed = time.perf_counter() - start
        print(f"RAG Ready: {elapsed:.3f}s")


    def switch(self, rag_path):
        rag_path = os.path.abspath(rag_path)
        if rag_path == self.current_path:
            print("RAG: Activated")
            return
        self.load(rag_path)


    def retrieve(self, question):
        if self.current_retriever is None:
            raise RuntimeError("Don't active RAG")
        return self.current_retriever.invoke(question)


# EMBEDDING
def load_embeddings():
    print("Embedding model loading...")
    start = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"Embedding: {time.perf_counter() - start:.2f}s")
    return embeddings


# LLM
def load_llm():
    print("LLM loading...")
    start = time.perf_counter()
    llm = Llama(model_path=chat_model, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
    print(f"[LLM] {time.perf_counter() - start:.2f}s")
    return llm
# ----------------------------


# loading
embeddings = load_embeddings()
rag_manager = RAGManager(embeddings)
llm = load_llm()


# FastAPI server
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# It saves the transmitted data to the RAG database (it is created if a RAG does not exist)
@app.post("/api/ragembed")
async def rag_embed(
    file: UploadFile = File(...),
    index_name: str = Form(...)
):
    # file save in /tmp cache
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output = langchain_embedding.process_files(tmp_path, index_name)
    if output == "not-supported-file":
      return {"status": output}
    elif output == "skipped":
      return {"status": output}
    elif output == "not-loaded":
      return {"status": output}
    elif output == "chroma-added-error":
      return {"status": output}
    elif output == "chroma-persist-error":
      return {"status": output}
    elif output == "successfully":
      return {"status": output}


# It delete the transmitted data to the RAG database
@app.post("/api/embeddelete")
async def embed_delete(file: UploadFile = File(...), rag_name: str = Form(...)):
    if not os.path.exists(rag_name):
      return {"status": "false", "output": "RAG not found !"}
    # file save in /tmp cache
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output = langchain_embedding.delete_removed_from_db(tmp_path, rag_name)
    return {"status": "true", "output": output}


# Ask AI with API
class AskAI(BaseModel):
    systemprompt: str
    question: str
    rag_name: str
    top_k: int
    ragsource: bool
@app.post("/api/askai")
def askai(req: AskAI):
  rag_manager.load(rag_folder+"/"+req.rag_name, req.top_k)  # loading RAG database
  output = langchain_ask.ask(llm, rag_manager, req.systemprompt, req.question, req.ragsource, os.path.basename(chat_model))
  return output


# Real-time response sending (only applicable for a suitable web interface)
@app.post("/api/askai_stream")
async def askai_stream(req: AskAI):
    rag_manager.load(rag_folder+"/"+req.rag_name, req.top_k)  # loading RAG database
    def event_stream():
        for chunk in langchain_ask.askai_stream(llm, rag_manager, req.systemprompt, req.question, req.ragsource, os.path.basename(chat_model)):
            yield chunk
    return StreamingResponse(event_stream(), media_type="text/plain")


# It retrieves a list of all RAG databases
@app.get("/api/listrag")
def listrag():
  if not os.path.exists(rag_folder):
    return {"status": "false", "error": "Not created RAG database"}
  raglist = []
  # it is being checked to see if it is a genuine RAG folder
  for ragl in os.listdir(rag_folder):  # It lists all RAG databases that have been created
    if os.path.exists(rag_folder+"/"+ragl+"/embedded_metadata.json"):
      raglist.append(ragl)
  return {"list": raglist}


# It retrieves a list of all RAG databases
class chcembed(BaseModel):
    rag_name: str
    hashvalue: str
@app.post("/api/checkembed")
def checkembed(req: chcembed):
  if not os.path.exists(rag_folder):
    return {"status": "false", "error": "Not created RAG database"}
  output = langchain_embedding.embed_check_hash(req.rag_name, req.hashvalue)
  if req.hashvalue in output:
    return {"status": "true"}
  else:
    return {"status": "false"}


# Delete selected RAG database
class dltrag(BaseModel):
    rag_name: str
@app.post("/api/deleterag")
def deleterag(req: dltrag):
    if not req.rag_name:
      return {"status": "false", "error": "RAG name was not entered !"}
    output = langchain_embedding.delete_ragdb(req.rag_name)
    if output["status"] == "false":  # if the process fails, it will be returned along with the necessary explanation
      return {"status": "false", "error": output["output"]}
    return {"status": "true", "output": output["output"]}


# Checking if the requested database exists
class ragcheck(BaseModel):
    rag_name: str
@app.post("/api/ragcheck")
def ragcheck(req: ragcheck):
  if os.path.exists(rag_folder+"/"+req.rag_name):  # check folder avalibility
    if os.path.exists(rag_folder+"/"+req.rag_name+"/embedded_metadata.json"):  # the folder is being checked to see if it is a RAG database
      return {"status": "true"}
    else:
      return {"status": "false"}
  else:
    return {"status": "false"}


# Check health API
@app.get("/api/health")
def healthcheck():
  return {"status": "true"}

