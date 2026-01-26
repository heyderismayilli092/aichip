import os
import time
import json
# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------- SETTINGS --------------------
MODEL_PATH = "/home/kali-sistem/99ELM01/ARDM/ai-models/Llama-3.2-8B-Instruct-Q3_K_M.gguf"  # yerel LLM
embedding_model = "/home/kali-sistem/99ELM01/ARDM/ai-models/all-mpnet-base-v2"
MAX_TOKENS = 2048  # LLM tarafından üretilen cevabın maksimum token sayısı

# -------------------- LLM AYARLARI (OPTİMİZE EDİLMİŞ) --------------------
#print("LLM yükleniyor...")
#llm = LlamaCpp(
#    streaming=True,
#    model_path=MODEL_PATH,
#    n_ctx=8192,           # Context window: uzun cevaplar ve RAG context için
#    n_threads=6,          # CPU thread sayısı
#    n_gpu_layers=0,       # GPU kullanımını kapattık (opsiyonel, RTX4050 ekleyebilirsin)
#    f16_kv=True,          # Half precision key/value => hız ve RAM optimizasyonu
#    max_tokens=MAX_TOKENS,# Üretilen cevabın maksimum token sayısı

    # ---------- TEKRAR ÖNLEME VE STABİL SAMPLING AYARLARI ----------
#    temperature=0.4,        # Düşük => daha tutarlı ve tekrar yapmayan cevap
#    top_k=30,               # Sampling top-k => token çeşitliliğini kontrol eder
#    top_p=0.9,              # Sampling top-p => nucleus sampling ile çeşitlilik
#    repeat_penalty=1.18,    # En önemli parametre: tekrar eden tokenları cezalandırır
#    repeat_last_n=256,      # Son üretilen 256 token üzerinde repeat_penalty uygular
#    stop=["###", "Context:", "Question:"],  # Stop token => prompt kesilince durur
#    verbose=False
#)
#print("LLM yüklendi.")



def ask(promptdata, question, ragdir, llm, top_k):
    now = time.time()
    ragdir = "ragdisk/"+ragdir
    if not os.path.exists(ragdir):
      return "Database folder not found !"
    # -------------------- START-UP: EMBEDDING VE DB --------------------
    print("RAG vektör veritabanı yükleniyor...")
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma(persist_directory=ragdir, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    
    # -------------------- RAG PROMPT --------------------
    prompt = ChatPromptTemplate.from_template("""
{prompt_data}

Question: {question}

Context:
{context}
""")

    # -------------------- RAG CHAIN --------------------
    rag_chain = (
        {
          "context": retriever,
          "question": RunnablePassthrough(),
          "prompt_data": RunnableLambda(lambda _: promptdata),
        }
        | prompt
        | llm
    )
    print("Vektör DB yüklendi.")

    # --- RETRIEVAL ---
    try:
        retrieved_docs = retriever.invoke(question)
    except Exception as e:
        return jsonify({"error": f"retriever failed: {e}"}), 500

    # --- RAG LLM CEVABI ---
    try:
        result = rag_chain.invoke(question, max_tokens=MAX_TOKENS)
        answer_text = str(result)
    except Exception as e:
        return jsonify({"error": f"LLM failed: {e}"}), 500

    # --- KAYNAKLARI TOPLA ---
    sources = []
    for d in retrieved_docs:
        meta = getattr(d, "metadata", None)
        if meta and "source" in meta:
            src = meta["source"]
        else:
            src = getattr(d, "source", None) or getattr(d, "page_content", "")[:80]
        if src and src not in sources:
            sources.append(src)
    return {"answer": answer_text, "sources": sources, "cached": False}



def askai_web(promptdata, question, ragdir, llm, top_k):
    now = time.time()
    ragdir = "ragdisk/"+ragdir
    if not os.path.exists(ragdir):
        return "Database folder not found !"

    # -------------------- START-UP: EMBEDDING VE DB --------------------
    print("RAG vektör veritabanı yükleniyor...")
    # Embedding modeli (offline / HF cache destekli)
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    # Chroma vektör veritabanı
    vectordb = Chroma(
        persist_directory=ragdir,
        embedding_function=embedding
    )
    # Retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": top_k}  # 🔹 kaç kaynak çekileceğini belirler
    )
    print("Vektör DB yüklendi.")

    # -------------------- RAG PROMPT --------------------
    prompt = ChatPromptTemplate.from_template("""
{prompt_data}

Question: {question}

Context:
{context}
    """)

    # -------------------- RAG CHAIN --------------------
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "prompt_data": RunnableLambda(lambda _: promptdata),
        }
        | prompt
        | llm  # 🔹 Streaming destekleyen LLM OLMALI
    )

    # -------------------- STREAMING GENERATOR --------------------
    """
    LLM cevabını parça parça üretir ve frontend'e anında gönderir.
    """

    # ---- KAYNAKLARI BAŞTA AL ----
    try:
        retrieved_docs = retriever.invoke(question)
    except Exception as e:
        yield f"\n[ERROR] retriever failed: {e}\n"
        return

    sources = []
    for d in retrieved_docs:
       meta = getattr(d, "metadata", None)
       if meta and "source" in meta:
          src = meta["source"]
       else:
          src = getattr(d, "source", None) or d.page_content[:80]
       if src and src not in sources:
           sources.append(src)

    # ---- LLM STREAM BAŞLAT ----
    try:
        for chunk in rag_chain.stream(question):
            """
            chunk:
            - LLM'in ürettiği token / text parçası
            - anında frontend'e gönderilir
            """
            yield chunk
    except Exception as e:
        yield f"\n[ERROR] LLM failed: {e}\n"
        return

    # ---- EN SON KAYNAKLARI GÖNDER ----
    yield "\n\n---\n"
    yield "Sources:\n"
    for s in sources:
        yield f"- {s}\n"




#output = ask("Based on the resources provided, summarize the points being discussed here.", ".", "sdr", llm, top_k)
#print(output["answer"])
#print(output["sources"])

