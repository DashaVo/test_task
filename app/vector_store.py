import pandas as pd
import re
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "embeddings/faiss_index.bin"
MAPPING_FILE = "embeddings/id_mapping.pkl"


def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(MAPPING_FILE, "rb") as f:
        id_map = pickle.load(f)
    return index, id_map

def search_similar(query, top_k=5, non_alcoholic=False, category_filter="cocktail"):
    index, id_map = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k * 2)

    results = []
    for i in I[0]:
        item = id_map[i]
        if non_alcoholic and item.get("alcoholic", "").lower() != "non alcoholic":
            continue
        if category_filter and category_filter.lower() not in item.get("category", "").lower():
            continue
        results.append(item)
        if len(results) == top_k:
            break

    return results

def search_similar_by_ingredient_reference(cocktail_name: str, top_k: int = 5):
    index, id_map = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)

    # знайди коктейль по імені
    target = next((item for item in id_map if item["Cocktail"].lower() == cocktail_name.lower()), None)
    if not target:
        return []

    ingredients = target.get("Ingredients")
    if not ingredients:
        return []

    query_vec = model.encode([ingredients]).astype("float32")
    D, I = index.search(query_vec, top_k + 1)  # +1 бо знайде і сам себе

    results = []
    for i in I[0]:
        match = id_map[i]
        if match["Cocktail"].lower() != cocktail_name.lower():
            results.append(match)
        if len(results) == top_k:
            break

    return results


def get_all_known_ingredients_from_csv(csv_path: str = "data/final_cocktails.csv") -> set[str]:
    df = pd.read_csv(csv_path)
    df.columns = [col.lower().strip() for col in df.columns]

    if "ingredients" not in df.columns:
        raise ValueError("Missing 'ingredients' column in CSV")

    words = set()

    for line in df["ingredients"].dropna():
        if isinstance(line, str):
            # розбиваємо складні назви на окремі слова
            for part in re.split(r"[,;/\|\-]", line):
                tokens = re.findall(r"\b[a-zA-Z]{3,}\b", part.lower())  # мін. 3 літери
                words.update(tokens)

    return words
