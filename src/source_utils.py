from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import hashlib
import os
import json


# SHA256 value calculate
def compute_sha256(path: str, block_size: int = 65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(block_size), b""):
            h.update(blk)
    return str(h.hexdigest())


# JSON metadata file is being loaded
def load_meta(path: str):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# JSON metadata file save
def save_meta(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# appropriate loader is selected based on the file format
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


# enables the sended file to be loaded
def safe_load_document(path: str):
    loader = choose_loader(path)
    if loader is None:
        return []
    try:
        return loader.load()
    except Exception as e:
        print(f"Load error: {path} -> {e}")
        return []

