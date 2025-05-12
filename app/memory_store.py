import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
MEMORY_INDEX_FILE = "embeddings/user_memory.index"
MEMORY_MAP_FILE = "embeddings/user_memory_map.pkl"

model = SentenceTransformer(MODEL_NAME)

def init_memory_index():
    if not os.path.exists(MEMORY_INDEX_FILE):
        index = faiss.IndexFlatL2(384)
        faiss.write_index(index, MEMORY_INDEX_FILE)
        with open(MEMORY_MAP_FILE, "wb") as f:
            pickle.dump([], f)

def save_user_memory(text: str):
    init_memory_index()
    vec = model.encode([text]).astype("float32")

    index = faiss.read_index(MEMORY_INDEX_FILE)
    index.add(vec)
    faiss.write_index(index, MEMORY_INDEX_FILE)

    with open(MEMORY_MAP_FILE, "rb") as f:
        memory = pickle.load(f)
    memory.append(text)
    with open(MEMORY_MAP_FILE, "wb") as f:
        pickle.dump(memory, f)

def get_user_memory() -> str:
    if not os.path.exists(MEMORY_MAP_FILE):
        return ""
    with open(MEMORY_MAP_FILE, "rb") as f:
        memory = pickle.load(f)
    return "\n".join(memory)
