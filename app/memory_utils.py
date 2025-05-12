from app.vector_store import get_all_known_ingredients_from_csv

def extract_ingredients_from_memory(memory_text: str) -> list[str]:
    known_ingredients = get_all_known_ingredients_from_csv()
    memory_text = memory_text.lower()

    found = []
    for ingr in known_ingredients:
        if ingr in memory_text:
            found.append(ingr)

    return list(set(found))

def detect_intent_fast(text: str) -> str | None:
    text = text.lower()

    # найперше — якщо є 'recommend' або подібні → це рекомендація
    if any(k in text for k in ["recommend", "suggest", "give me", "show me"]):
        return "request_recommendation"

    # потім — якщо це запит про улюблене
    if "what" in text and any(k in text for k in ["like", "favourite", "favorite", "remember", "ingredient"]):
        return "ask_preferences"

    # потім — якщо користувач ділиться вподобаннями
    if any(k in text for k in ["i like", "i love", "i prefer", "i enjoy", "my favourite", "my favorite"]) and "what" not in text:
        return "remember_preference"

    # потім — якщо посилання на коктейль
    if any(k in text for k in ["similar to", "based on", "reminds me of"]):
        return "reference_drink"

    return None
