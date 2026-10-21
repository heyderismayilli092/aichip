import os
import sys
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
import mimetypes
import magic
import source_utils  # provides the necessary preliminary functions for processing the given resources
# Langchain frameworks
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ---------- CONFIG ----------
with open("config.json", "r") as configfile:
  data = json.load(configfile)
EMBED_MODEL = data["embed-model"]  # HuggingFace embedding model
rag_folder = data["rag"]["rag-folder"]  # RAG database directory
chunk_size = data["rag"]["chunk-size"]
chunk_overlap = data["rag"]["chunk-overlap"]
# ----------------------------


# Deleting a file embedded in a RAG database from the database
def delete_removed_from_rag(embedfile: str, db_dir: str):
    db_dir = rag_folder+"/"+db_dir
    if not os.path.exists(db_dir):
      return {"status": "false", "output": "Database folder not found !"}
    # Initialize embedding + db (embedding_function passed to Chroma when writing)
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=db_dir, embedding_function=embedding)
    # load json embedding metadata
    meta = source_utils.load_meta(db_dir+"/embedded_metadata.json")
    embedded = []  # a list of previously embedded files is being added.
    # detect removed files and add list
    for file in meta:
      embedded.append(file)
    name = os.path.basename(embedfile)
    # previously buried but later abandoned resources are being removed from the database
    if not name in embedded:
        return {"status": "false", "output": "No such file was found in the database!" }

    print(f"File removed in source: {name}  -> Remove from ChromaDB")
    try:
        db.delete(where={"source": name})
    except Exception as e:
        return {"status": "false", "output": f"Chroma delete error ({os.path.basename(name)}): {e}"}
    del meta[name]  # when a resource is deleted from the database, the information belonging to that resource is also deleted from the JSON file
    with open(db_dir+"/embedded_metadata.json", "w", encoding="utf-8") as f:  # saving the updated JSON file
      json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"status": "true", "output": "Successfully delete file in database"}


# Deleting a file embedded in a RAG database from the database
def delete_ragdb(rag_name: str):
  rag_name = "ragdisk/"+rag_name
  if not os.path.exists(rag_name):  # checking RAG database avalibility
    return {"status": "false", "output": "RAG folder not found !"}
  os.system(f"rm -r {rag_name}")  # deleting RAG folder
  return {"status": "true", "output": "The RAG database has been successfully deleted"}


# Embedding the submitted file into the RAG database
def ragembed(embedfile: str, db_dir: str):
    db_dir = rag_folder+"/"+db_dir
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

    meta = source_utils.load_meta(db_dir+"/embedded_metadata.json")
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    mtime = int(st.st_mtime)
    fsize = st.st_size
    rec = meta.get(fname)
    if not rec is None:
      # Fast checks: if mtime or size differs, we should inspect content
      if rec.get("mtime") != mtime or rec.get("filesize") != fsize:
        # compute quick sha to be sure (we could compute sha only after loading, but do now)
        sha = source_utils.compute_sha256(fullpath)
        if sha != rec.get("sha256"):
            # We will collect all chunks across files so that we can show overall progress.
            # But to avoid reading everything in memory we process file-by-file and add to DB gradually.
            print(f"\nBeing processed: {os.path.basename(fname)}")

            # compute sha now (ensure we store correct sha)
            sha = source_utils.compute_sha256(fullpath)

            # load document (only now)
            docs = source_utils.safe_load_document(fullpath)
            if not docs:
                return "not-loaded"

            # chunk
            chunks = splitter.split_documents(docs)
            chunk_count = len(chunks)
            print(f"Chunk number: {chunk_count}")

    # compute sha now (ensure we store correct sha)
    sha = source_utils.compute_sha256(fullpath)
    # load document (only now)
    docs = source_utils.safe_load_document(fullpath)
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
    source_utils.save_meta(db_dir+"/embedded_metadata.json", meta)
    return f"successfully"


# Source embed check with hash value
def ragembed_check_hash(db_dir, sha256: str) -> bool:
  db_dir = rag_folder+"/"+db_dir
  shavalues = []  # hash values list
  with open(db_dir+"/embedded_metadata.json", "r") as jsdata:  # load the embedding metadata file
    data = json.load(jsdata)
  for embedfile in data.keys():  # list the files name
    shavalues.append(data[embedfile]["sha256"])  # sha valuess add in list
  return shavalues

