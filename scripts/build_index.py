import os
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

CSV_PATH = "../data/final_cocktails.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "../embeddings/faiss_index.bin"
MAPPING_FILE = "../embeddings/id_mapping.pkl"

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [col.lower().strip() for col in df.columns]

    if "ingredients" not in df.columns or "name" not in df.columns:
        raise ValueError("CSV must contain 'name' and 'ingredients' columns")

    df = df.dropna(subset=["name", "ingredients"])
    df = df[df["ingredients"].apply(lambda x: isinstance(x, str))]

    # Заповнити відсутні колонки, якщо вони не існують
    for col in ["alcoholic", "category", "glasstype", "instructions"]:
        if col not in df.columns:
            df[col] = ""

    # Побудова тексту для векторизації
    df["text"] = df["name"] + " - " + df["ingredients"]

    return df

def build_faiss_index(df, model_name=EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index

def save_index(index, df):
    os.makedirs("../embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    metadata = df[[
        "name",
        "ingredients",
        "alcoholic",
        "category",
        "glasstype",
        "instructions"
    ]].to_dict(orient="records")

    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Index and metadata saved.")

def main():
    print("📥 Loading dataset...")
    df = load_dataset(CSV_PATH)

    print(f"📦 Loaded {len(df)} cocktail entries.")
    print("🔗 Building embeddings and FAISS index...")
    index = build_faiss_index(df)

    print("💾 Saving index and metadata...")
    save_index(index, df)

    print("✅ Done.")

if __name__ == "__main__":
    main()
