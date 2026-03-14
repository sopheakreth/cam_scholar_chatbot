# src/demo.py
import streamlit as st
from chatbot import chatbot_response, get_response, intents, recommend_university

st.set_page_config(page_title="CAM Scholar ChatBot", layout="centered")
st.title("CAM Scholar ChatBot Demo")
st.markdown("Step-by-step chatbot to recommend universities in Cambodia.")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""


# Function to handle sending messages
def send_message():
    user_msg = st.session_state.user_input.strip()
    if user_msg:
        st.session_state.chat_history.append("")
        st.session_state.chat_history.append(f"**You:** {user_msg}")

        # Chatbot logic
        ints = chatbot_response(user_msg)
        tag = ints[0]['intent']

        if tag in ["ask_major", "ask_grade", "ask_location"]:
            bot_msg = get_response([{"intent": tag}], intents)
        elif tag == "show_recommendation":
            results = recommend_university()
            if results:
                bot_msg = "Here are some universities that match your criteria:\n\n"
                for u in results:
                    bot_msg += f"- **{u['university']}** in {u['location']} with a scholarship of {u['scholarship']}% (Match Score: {u['score']}%)\n"
            #     bot_msg = "\n".join(
            #         [f"{u['university']} | {u['major']} | {u['location']} | Scholarship: {u['scholarship']}%" for u in
            #          results]
            #     )
            else:
                bot_msg = "Sorry, no universities match your criteria."
        else:
            bot_msg = get_response(ints, intents)

        st.session_state.chat_history.append(f"**Bot:** {bot_msg}")

        # Clear input box
        st.session_state.user_input = ""


# Input box with Enter key trigger
st.text_input("Type your message:", key="user_input", on_change=send_message, placeholder="Type here...")

# Scrollable chat area
chat_display = "\n\n".join(st.session_state.chat_history)
st.markdown(
    f"<div style='height:500px; overflow-y:auto; border:1px solid #ddd; border-radius: 10px; padding:10px'>{chat_display}</div>",
    unsafe_allow_html=True
)
