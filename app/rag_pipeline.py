from app.memory_utils import extract_ingredients_from_memory
from app.vector_store import search_similar
from app.hf_llm import generate_answer

def is_non_alcoholic_question(q: str) -> bool:
    return "non-alcoholic" in q.lower() or "non alcoholic" in q.lower()

def answer_with_rag(question: str, memory_context: str = "", top_k: int = 5) -> str:

    print(f"[USER MEMORY]: {memory_context}")
    ingredients = extract_ingredients_from_memory(memory_context)
    print(f"[EXTRACTED INGREDIENTS]: {ingredients}")


    if ingredients:
        query = "cocktails with " + ", ".join(ingredients)
    else:
        query = question


    non_alc = is_non_alcoholic_question(question)
    category = detect_category_filter(question)


    results = search_similar(query, top_k=top_k, non_alcoholic=non_alc, category_filter=category)


    cocktail_context = "\n".join(
        f"{r['name']}: {r['ingredients']}" for r in results
    )


    full_context = (memory_context + "\n" + cocktail_context).strip()


    return generate_answer(full_context, question)

def detect_category_filter(question: str) -> str:
    q = question.lower()

    if "shot" in q:
        return "shot"
    if "beer" in q:
        return "beer"
    if "tea" in q or "coffee" in q:
        return "coffee / tea"
    if "soft drink" in q or "soda" in q:
        return "soft drink"
    if "punch" in q or "party drink" in q:
        return "punch / party drink"
    if "milkshake" in q or "shake" in q:
        return "shake"
    if "cocoa" in q or "chocolate" in q:
        return "cocoa"
    if "liqueur" in q or "homemade" in q:
        return "homemade liqueur"
    if "ordinary drink" in q:
        return "ordinary drink"
    if "non-alcoholic" in q:
        return ""  # allow all, will be filtered separately
    if "unknown" in q:
        return "other / unknown"

    return "cocktail"  # default fallback

def is_reference_to_existing_cocktail(question: str) -> str | None:
    triggers = ["similar to", "like", "based on"]
    for trigger in triggers:
        if trigger in question.lower():
            return question.lower().split(trigger)[-1].strip()
    return None

