#!/usr/bin/env python3

import os
import sys
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
import mimetypes
import magic
# Langchain frameworks
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------- CONFIG ----------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "/home/kali-sistem/99ELM01/ARDM/ai-models/all-mpnet-base-v2"  # HuggingFace embedding model
# ----------------------------


# --------FUNCTIONS-----------
def compute_sha256(path: str, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(block_size), b""):
            h.update(blk)
    return h.hexdigest()


def load_meta(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_meta(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def choose_loader(path: str):
    lower = path.lower()
    if lower.endswith(".pdf"):
        return PyPDFLoader(path)
    if lower.endswith(".docx"):
        return Docx2txtLoader(path)
    if lower.endswith(".txt"):
        return TextLoader(path, encoding="utf-8")
    if lower.endswith(".md"):
        return UnstructuredMarkdownLoader(path)
    if lower.endswith(".html") or lower.endswith(".htm"):
        return UnstructuredHTMLLoader(path)
    return None


def safe_load_document(path: str):
    loader = choose_loader(path)
    if loader is None:
        return []
    try:
        return loader.load()
    except Exception as e:
        print(f"Load error: {path} -> {e}")
        return []
        
#-----------------------------------------------------


def delete_removed_from_db(embedfile: str, db_dir: str):
    db_dir = "ragdisk/"+db_dir
    if not os.path.exists(db_dir):
      return "Database folder not found !"
    # Initialize embedding + db (embedding_function passed to Chroma when writing)
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=db_dir, embedding_function=embedding)
    # load json embedding metadata
    meta = load_meta(db_dir+"/embedded_metadata.json")
    embedded = []  # a list of previously embedded files is being added.
    # detect removed files and add list
    for file in meta:
      embedded.append(file)
    name = os.path.basename(embedfile)
    # previously buried but later abandoned resources are being removed from the database
    if not name in embedded:
        return "No such file was found in the database!"

    print(f"File removed in source: {name}  -> Remove from ChromaDB")
    try:
        db.delete(where={"source": name})
    except Exception as e:
        return f"Chroma delete error ({os.path.basename(name)}): {e}"
    del meta[name]  # when a resource is deleted from the database, the information belonging to that resource is also deleted from the JSON file
    with open(db_dir+"/embedded_metadata.json", "w", encoding="utf-8") as f:  # saving the updated JSON file
      json.dump(meta, f, indent=2, ensure_ascii=False)
    return "Successfully delete file in database"


def process_files(embedfile: str, db_dir: str):
    db_dir = "ragdisk/"+db_dir
    os.makedirs(db_dir, exist_ok=True)

    SUPPORTED_EXTS = (".pdf", ".txt", ".docx", ".md", ".html")  # supported embed format list
    mime = magic.Magic(mime=True)
    mimetyp = mime.from_file(embedfile)
    extension = mimetypes.guess_extension(mimetyp)
    if extension not in SUPPORTED_EXTS:
      return "not-supported-file"

    # Initialize embedding + db (embedding_function passed to Chroma when writing)
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=db_dir, embedding_function=embedding)

    meta = load_meta(db_dir+"/embedded_metadata.json")
    embedded = []  # a list of previously embedded files is being added.
    for file in meta:
      embedded.append(file)
    if os.path.basename(embedfile) in embedded:
        return "skipped"

    # Decide which files need processing
    fullpath = embedfile
    fname = os.path.basename(embedfile)
    st = os.stat(fullpath)

    # Load and process each file one by one (don't load all at once)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    mtime = int(st.st_mtime)
    fsize = st.st_size
    rec = meta.get(fname)
    if not rec is None:
      # Fast checks: if mtime or size differs, we should inspect content
      if rec.get("mtime") != mtime or rec.get("filesize") != fsize:
        # compute quick sha to be sure (we could compute sha only after loading, but do now)
        sha = compute_sha256(fullpath)
        if sha != rec.get("sha256"):
            # We will collect all chunks across files so that we can show overall progress.
            # But to avoid reading everything in memory we process file-by-file and add to DB gradually.
            print(f"\nBeing processed: {os.path.basename(fname)}")

            # compute sha now (ensure we store correct sha)
            sha = compute_sha256(fullpath)

            # load document (only now)
            docs = safe_load_document(fullpath)
            if not docs:
                return "not-loaded"

            # chunk
            chunks = splitter.split_documents(docs)
            chunk_count = len(chunks)
            print(f"Chunk number: {chunk_count}")

    # compute sha now (ensure we store correct sha)
    sha = compute_sha256(fullpath)
    # load document (only now)
    docs = safe_load_document(fullpath)
    if not docs:
        return f"not-loaded"
    # chunk
    chunks = splitter.split_documents(docs)
    chunk_count = len(chunks)
    print(f"Chunk number: {chunk_count}")
    # annotate metadata for each chunk and add progressively
    pbar = tqdm(chunks, desc=f"Embedding {fname}", unit="chunk")
    added = 0
    for idx, chunk in enumerate(pbar):
        # Ensure each chunk has metadata with source and file_hash
        if not hasattr(chunk, "metadata") or chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata["source"] = fname
        chunk.metadata["sha256"] = sha
        chunk.metadata["chunk_index"] = idx

        try:
            db.add_documents([chunk])
            added += 1
        except Exception as e:
            return "chroma-added-error"
            pbar.close()
            break
        pbar.update(0)  # tqdm requires manual update only if not iterating normally

    pbar.close()
    # after file processed, persist DB (safe) and update metadata record
    try:
        db.persist()
    except Exception as e:
        return f"chroma-persist-error"

    meta[fname] = {
        "sha256": sha,
        "mtime": int(os.stat(fullpath).st_mtime),
        "filesize": os.stat(fullpath).st_size,
        "chunk_count": chunk_count
    }
    # Save metadata JSON
    save_meta(db_dir+"/embedded_metadata.json", meta)
    return f"successfully"

