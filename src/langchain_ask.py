import os
import time
import json
# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
#from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
#from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings

# loading config file
with open("config.json", "r") as configfile:  # loading config file
  data = json.load(configfile)
max_tokens = data["max-tokens"]


# prompt mechanism
PROMPT = PromptTemplate.from_template(
    """{systemprompt}

Context:
{context}

Question:
{question}

Answer:"""
)

# context
def build_context(documents):
    return "\n\n".join(document.page_content for document in documents)


def ask(llm, rag_manager, systemprompt, question, ragsource, chatmodel):
    # RETRIEVAL
    start = time.perf_counter()
    documents = rag_manager.retrieve(question)
    retrieval_time = (time.perf_counter() - start)
    print(f"[Retrieval] {retrieval_time:.3f}s")
    print(f"[Documents] {len(documents)}")

    # SOURCES
    if ragsource == True:
      sources = []
      for document in documents:
        metadata = getattr(document, "metadata", {})
        source = metadata.get("source")
        if source and source not in sources:
            sources.append(source)

    # CONTEXT
    context = build_context(documents)
    print(f"[Context] {len(context)} characters")

    # PROMPT
    prompt = PROMPT.format(systemprompt=systemprompt, context=context, question=question)

    # GENERATION
    print("Generating...")
    generation_start = time.perf_counter()
    first_token_time = None
    full_answer = []
    token_count = 0

    stream = llm(prompt, max_tokens=max_tokens, temperature=0.1, stop=["Question:"], stream=False)  # streaming
    generation_time = (time.perf_counter() - generation_start)

    # OUTPUT
    if ragsource == True:
      return {"answer": stream, "chatmodel": chatmodel, "sources": sources, "generation_time": str(generation_time)}
    else:
      return {"answer": stream, "chatmodel": chatmodel, "generation_time": str(generation_time)}



def askai_stream(llm, rag_manager, systemprompt, question, ragsource, chatmodel):
    # RETRIEVAL
    start = time.perf_counter()
    documents = rag_manager.retrieve(question)
    retrieval_time = (time.perf_counter() - start)
    print(f"Retrieval: {retrieval_time:.3f}s")
    print(f"Documents: {len(documents)}")

    # SOURCES
    if ragsource == True:
      sources = []
      for document in documents:
        metadata = getattr(document, "metadata", {})
        source = metadata.get("source")
        if source and source not in sources:
            sources.append(source)

    # CONTEXT
    context = build_context(documents)
    print(f"Context: {len(context)} characters")

    # PROMPT
    prompt = PROMPT.format(systemprompt=systemprompt, context=context, question=question)

    # GENERATION
    print("Generating...")
    generation_start = time.perf_counter()
    first_token_time = None
    token_count = 0

    stream = llm(prompt, max_tokens=max_tokens, temperature=0.1, stop=["Question:"], stream=True)  # streaming

    # OUTPUT
    for chunk in stream:
        text = chunk["choices"][0]["text"]
        if not text:
            continue
        if first_token_time is None:
            first_token_time = (time.perf_counter()- generation_start)
        token_count += 1
        yield text
    #generation_time = (time.perf_counter() - generation_start)
    #yield {"generation_time": str(generation_time), "chatmodel": chatmodel}

    yield "\n\n---\n"
    yield "Sources:\n"
    for source in sources:
      yield f"- {source}\n"

