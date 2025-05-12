# 🍸 Cocktail Advisor Chat

An intelligent, LLM-powered chat application that recommends cocktails based on user preferences, memory, and semantic search through a cocktail dataset using FAISS and RAG (Retrieval-Augmented Generation).

---

## 🔍 Features

- 💬 Chat with an AI bartender
- 🧠 Personalized memory — remembers what you like (e.g., "I like gin")
- 🧾 Retrieval-Augmented Generation using a cocktail knowledge base
- 🧊 FAISS vector search over cocktail ingredients
- 🌐 FastAPI backend with REST API
- 🎛️ Streamlit chat UI
- 🤖 Hugging Face LLM for answer generation & intent classification

---

## 🧪 Example questions

### 📚 Knowledge-based:
- What are the 5 cocktails containing lemon?
- What are the 5 non-alcoholic cocktails containing sugar?
- What are my favourite ingredients?

### 🧠 Advisor-based:
- Recommend 5 cocktails that contain my favourite ingredients
- Recommend a cocktail similar to “Hot Creamy Bush”

---

## 🚀 Quickstart

### 1. Clone the repo


git clone https://github.com/DashaVo/test_task.git
cd test_task

### 2. Create and activate environment

conda create -n test_task python=3.10
conda activate test_task

## 3. Install dependencies

pip install -r requirements.txt

### 4. Add Hugging Face API key

Create .env file in the project root:
HF_API_KEY=your_huggingface_inference_token

### 5. Download the dataset

Download final_cocktails.csv and place it in:
data/final_cocktails.csv

### 6. Build the FAISS index
python scripts/build_index.py

### 7. Run FastAPI backend

uvicorn app.main:app --reload
API Docs available at: http://localhost:8000/docs

### 8. Run Streamlit frontend

streamlit run streamlit_app/chat_ui.py
Access the chat at: http://localhost:8501

```bash
