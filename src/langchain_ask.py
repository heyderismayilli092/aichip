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
models_path = data["models-path"]
chat_model = os.path.join(models_path, data["llm"]["model"])
max_tokens = data["llm"]["max-tokens"]
temperature = data["llm"]["temperature"]
chunk_size = data["rag"]["chunk-size"]
chunk_overlap = data["rag"]["chunk-overlap"]
sum_max_tokens = data["summary"]["max-tokens"]
sum_temperature = data["summary"]["temperature"]


# context
def build_context(documents):
    return "\n\n".join(document.page_content for document in documents)


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

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager=None, **kwargs,) -> str:
        print("="*15)
        print("MODEL CALL DEBUG:")
        print("="*15)
        print("Prompt length:", len(prompt))
        print("Prompt preview:")
        print(repr(prompt[:1000]))
        print("Stop:", stop)
        print("Max tokens:", self.max_tokens)
        print("Temperature:", self.temperature)

        response = self.client(prompt, max_tokens=self.max_tokens, temperature=self.temperature, stop=stop, **kwargs,)
        print("="*15)
        print("RAW MODEL RESPONSE:")
        print("="*15)
        print("Finish reason:", response["choices"][0].get("finish_reason"))
        print("Output length:", len(response["choices"][0]["text"]))
        print("Output preview:", repr(response["choices"][0]["text"][:500]))
        print("=="*20, "\n")
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


# a prompt is sent to the loaded raw model, and a response is generated
def ask(llm, question, systemprompt=None, event_callback=None):
    # takes the events and adds them to the monitoring queue
    def eventlog(event, **data):
        if event_callback:
            event_callback({"event": event, "timestamp": time.time(), **data})

    try:
        eventlog("generation_start")
        generation_start = time.perf_counter()

        if systemprompt:
            rolelist = [
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": question}
            ]
        else:
            rolelist = [
                {"role": "user", "content": question}
            ]

        eventlog("prompt_prepared", message_count=len(rolelist))
        eventlog("llm_start")

        # generating response
        response = llm.create_chat_completion(
            messages=rolelist,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        finish_reason = None
        usage = None
        model = None
        created = None

        answer = ""
        rawresponse = []

        for chunk in response:
            rawresponse.append(chunk)
            choice = chunk["choices"][0]
            token = choice.get("delta", {}).get("content", "")

            if token:
                answer += token
                eventlog("llm_token", token=token)

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            # stream metadata
            model = chunk.get("model", model)
            created = chunk.get("created", created)
            usage = chunk.get("usage", usage)

        generation_time = time.perf_counter() - generation_start  # generation time

        # reconstruct a response-like dictionary
        final_response = {
            "id": rawresponse[0].get("id") if rawresponse else None,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": finish_reason
                }
            ],
            "usage": usage
        }

        eventlog("generation_end", generation_time=generation_time, finish_reason=finish_reason)
        print("Generated response:", generation_time)
        return {"answer": final_response, "chatmodel": os.path.basename(chat_model), "generation_time": str(generation_time)}
    except Exception as e:
        eventlog("generation_error", error=str(e))
        raise


def ask_stream(llm, question, systemprompt=None, event_callback=None):
    # takes the events and adds them to the monitoring queue
    def eventlog(event, **data):
        if event_callback:
            event_callback({"event": event, "timestamp": time.time(), **data})

    try:
        eventlog("generation_start")
        generation_start = time.perf_counter()

        if systemprompt:
            rolelist = [
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": question}
            ]
        else:
            rolelist = [
                {"role": "user", "content": question}
            ]

        eventlog("prompt_prepared", message_count=len(rolelist))
        eventlog("llm_start")

        # generating response
        response = llm.create_chat_completion(
            messages=rolelist,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        finish_reason = None
        usage = None
        model = None
        created = None

        for chunk in response:
            choice = chunk["choices"][0]
            token = choice.get("delta", {}).get("content", "")

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            # stream metadata
            model = chunk.get("model", model)
            created = chunk.get("created", created)
            usage = chunk.get("usage", usage)

            # realtime send tokens
            if token:
                eventlog("llm_token", token=token)
                yield token

        generation_time = time.perf_counter() - generation_start  # generation time
        eventlog("llm_end", generation_time=generation_time, finish_reason=finish_reason)
        eventlog("generation_end")
        print("Generated response:", generation_time)
        yield "-- chat_model: "+os.path.basename(chat_model)
        yield "-- generation_time: "+str(generation_time)
    except Exception as e:
        eventlog("generation_error", error=str(e))
        raise


# a response to the prompt is generated using a RAG database
def ragask(llm, rag_manager, systemprompt, question, ragsource):
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
      return {"answer": stream, "chatmodel": os.path.basename(chat_model), "sources": sources, "generation_time": str(generation_time)}
    else:
      return {"answer": stream, "chatmodel": os.path.basename(chat_model), "generation_time": str(generation_time)}


# a response to the prompt is generated using a RAG database (the answer is produced and displayed in real time)
def ragask_stream(llm, rag_manager, systemprompt, question, ragsource):
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


# a summary of the submitted document is generated and displayed to the user
def summary(llm, chaintype, sourcedoc, mapprompt=None, reduceprompt=None, stuffprompt=None):
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
    # 'Map Reduce' method
    if chaintype == "map_reduce":
        print("Mode: map_reduce")
        if mapprompt:
            # Map Stage (Instruction used while summarizing each chunk individually)
            map_template = """{mapprompt}
{text}
"""
            map_prompt = PromptTemplate(template=map_template, input_variables=["text"]).partial(mapprompt=mapprompt)  # relevant system prompt is provided for the summarization of each part
        if reduceprompt:
            # B) Reduce Stage (Instruction used while combining chunk summaries into a final report)
            reduce_template = """{reduceprompt}
{text}
"""
            reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["text"]).partial(reduceprompt=reduceprompt)  # a system prompt is given to generate a general summary of all the generated summaries

        if mapprompt != None or reduceprompt != None:
            summarychain = load_summarize_chain(langchain_llm, chain_type="map_reduce", verbose=False)
        else:
            summarychain = load_summarize_chain(langchain_llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=reduce_prompt, verbose=False)
        output = summarychain.invoke(splitted_source)

    # 'Stuff' method
    elif chaintype == "stuff":
        print("Mode: stuff")
        # if the user has not provided an additional system prompt for summarization, no prompt is sent
        if stuffprompt:
            stuff_template = """{stuffprompt}
{text}"""
            stuff_prompt = PromptTemplate(template=stuff_template, input_variables=["text"]).partial(stuffprompt=stuffprompt)  # a system prompt is given to generate a general summary of all the generated summaries
            summarychain = load_summarize_chain(langchain_llm, chain_type="stuff", prompt=stuff_prompt, verbose=False)
        else:
            summarychain = load_summarize_chain(langchain_llm, chain_type="stuff", verbose=False)
        output = summarychain.invoke(splitted_source)

    return output

