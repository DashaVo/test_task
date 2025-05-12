from fastapi import FastAPI
from app.models.schemas import ChatRequest, ChatResponse
from app.memory_store import save_user_memory, get_user_memory
from app.memory_utils import detect_intent_fast, extract_ingredients_from_memory
from app.rag_pipeline import answer_with_rag
from app.hf_llm import classify_intent
import uvicorn

app = FastAPI(title="Cocktail Advisor Chat API")

@app.get("/memory/debug")
def debug_memory():
    return {"memory": get_user_memory()}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.question.strip()

    # Step 1: fast keyword-based intent detection
    intent = detect_intent_fast(text)

    # Step 2: fallback to LLM intent classification only if unclear
    if not intent:
        try:
            intent = classify_intent(text)
            print(f"[LLM INTENT] {intent}")
        except Exception as e:
            print(f"[INTENT ERROR] {e}")
            intent = "chat"
    else:
        print(f"[FAST INTENT] {intent}")

    # Handle intent
    if intent == "ask_preferences":
        memory = get_user_memory()
        print(f"[MEMORY RAW] {memory}")  # 🪵 це найважливіше
        ingredients = extract_ingredients_from_memory(memory)
        print(f"[INGREDIENTS EXTRACTED] {ingredients}")

        if ingredients:
            return ChatResponse(answer=f"Based on our chat so far, you like: {', '.join(ingredients)}.")
        else:
            return ChatResponse(answer="I don't know your preferences yet.")

    elif intent == "remember_preference":
        save_user_memory(text)
        return ChatResponse(answer="Got it! I've saved your favorite ingredients.")

    else:
        memory_context = get_user_memory()
        answer = answer_with_rag(text, memory_context=memory_context)
        return ChatResponse(answer=answer)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
