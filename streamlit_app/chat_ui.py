import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

# Ініціалізація історії чату
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Cocktail Advisor Chat", page_icon="🍸")
st.title("🍸 Cocktail Advisor Chat")

# Вивід історії повідомлень
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ввід користувача
if user_input := st.chat_input("Ask me anything about cocktails..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Mixing your drink..."):
            try:
                response = requests.post(API_URL, json={"question": user_input})
                response.raise_for_status()
                answer = response.json().get("answer", "Hmm... I didn’t get that.")
            except requests.exceptions.RequestException as e:
                answer = "😓 Sorry, something went wrong. Please try again later."
                print(f"[API ERROR] {e}")
            except Exception as e:
                answer = "❌ Unexpected error occurred."
                print(f"[UNHANDLED ERROR] {e}")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
