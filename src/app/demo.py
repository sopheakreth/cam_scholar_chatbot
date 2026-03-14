# src/demo.py
import streamlit as st
from chatbot import chatbot_response, get_response, intents, recommend_university

def set_bg_gradient():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #FADAFF 50%, #BAC6FF 100%);
            background-attachment: fixed;
        }
        
        /* Styling the Title (h1) */
        h1 {
            color: #1e40af !important; /* A deep blue for better contrast */
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
            text-align: center;
        }
        
        # .stMarkdown p {
        #     text-align: center;
        # }

        /* Optional: Styling subheaders (h2, h3) to be green */
        h2, h3 {
            color: #065f46 !important; /* A dark forest green */
        }

        /* Ensures the header doesn't have a solid white block */
        header {
            background: rgba(0,0,0,0) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_gradient()

st.set_page_config(page_title="CAM Scholar ChatBot", layout="centered")
st.title("CAM Scholar ChatBot")
st.markdown(
    "<div style='text-align: center; margin-bottom: 20px;'>Hi! I can help you choose a university in Cambodia. Let's start get started.</div>",
            unsafe_allow_html=True
)

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
                    entrance_exam = "required" if u['entrance_exam'] == "Yes" else "not required"
                    bot_msg += f"- **{u['university']}** in {u['location']} with a scholarship of {u['scholarship']}%. The total quota is **{u['quota']}** and an entrance exam is **{entrance_exam}** (Match Score: {u['score']}%)\n"
            else:
                bot_msg = "Sorry, no universities match your criteria."
        else:
            bot_msg = get_response(ints, intents)

        st.session_state.chat_history.append(f"**Bot:** {bot_msg}")

        # Clear input box
        st.session_state.user_input = ""


# Input box with Enter key trigger
st.text_input("Type your message", key="user_input", on_change=send_message, placeholder="Type here...", label_visibility="collapsed")

# Scrollable chat area
chat_display = "\n\n".join(st.session_state.chat_history)
st.markdown(
    f"<div style='height: 600px; overflow-y:auto; border:1px solid #ddd; border-radius: 10px; padding:10px; background-color: #FFFFFF;'>{chat_display}</div>",
    unsafe_allow_html=True
)
