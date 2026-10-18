import os
import time
import json
import source_utils  # provides the necessary preliminary functions for processing the given resources
# LangChain frameworks
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict
from typing import Any

# loading config file
with open("config.json", "r") as configfile:  # loading config file
  data = json.load(configfile)
max_tokens = data["llm"]["max-tokens"]
temperature = data["llm"]["temperature"]
chunk_size = data["rag"]["chunk-size"]
chunk_overlap = data["rag"]["chunk-overlap"]
chat_model = data["chat-model"]
sum_max_tokens = data["summary"]["max-tokens"]
sum_temperature = data["summary"]["temperature"]


# structure for adapting a raw-loaded model into LangChain
class ExistingLlamaLLM(LLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Any
    max_tokens: int = 512
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "existing-llama-cpp"

    @property
    def _identifying_params(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager=None, **kwargs: Any,) -> str:
        response = self.client(prompt, max_tokens=self.max_tokens, temperature=self.temperature, stop=stop, **kwargs,)
        return response["choices"][0]["text"]


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

    stream = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["Question:"], stream=False)  # streaming
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

    stream = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["Question:"], stream=True)  # streaming

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

    if ragsource == True:
      yield "\n\n---\n"
      yield "Sources:\n"
      for source in sources:
        yield f"- {source}\n"


def summary(llm, chaintype, mapprompt, reduceprompt, sourcedoc):
    document = source_utils.safe_load_document(sourcedoc)  # uploaded file is being loaded
    # the document is being broken down into parts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_source = text_splitter.split_documents(document)
    # separated parts are examined and summarized by the model
    langchain_llm = ExistingLlamaLLM(
        client=llm,
        max_tokens=sum_max_tokens,
        temperature=sum_temperature,
    )
    # Map Stage (Instruction used while summarizing each chunk individually)
    map_template = """{mapprompt}
{text}
"""
    map_prompt = PromptTemplate(template=map_template, input_variables=["text"]).partial(mapprompt=mapprompt)  # relevant system prompt is provided for the summarization of each part

    # B) Reduce Stage (Instruction used while combining chunk summaries into a final report)
    reduce_template = """{reduceprompt}
{text}
"""
    reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["text"]).partial(reduceprompt=reduceprompt)  # a system prompt is given to generate a general summary of all the generated summaries
    summarychain = load_summarize_chain(langchain_llm, chain_type=chaintype, map_prompt=map_prompt, combine_prompt=reduce_prompt, verbose=False)
    output = summarychain.invoke(splitted_source)
    return output

