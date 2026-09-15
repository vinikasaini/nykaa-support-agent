import streamlit as st
import sys
from pathlib import Path

# Make sure the project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import agent



# PAGE CONFIGURATION
st.set_page_config(
    page_title="Nykaa Support Assistant",
    page_icon="🛍️",
    layout="centered"
)



# CUSTOM CSS
st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 25px;
    }

    .stChatMessage {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# HEADER
st.markdown(
    '<div class="main-title">🛍️ Nykaa Support Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about orders, returns, refunds, delivery, payments and more.'
    '</div>',
    unsafe_allow_html=True
)



# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []



# DISPLAY PREVIOUS MESSAGES
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])



# HELPER FUNCTION
def clean_answer(answer):

    if not answer:
        return "Sorry, I could not generate a response."

    # Remove mock LLM technical labels
    answer = answer.replace(
        "MOCK_LLM ANSWER",
        ""
    )

    answer = answer.replace(
        "Answer based only on retrieved context:",
        ""
    )

    return answer.strip()



# CHAT INPUT
query = st.chat_input(
    "Type your question here..."
)


if query:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    
    # CALL YOUR EXISTING AGENT

    try:

        result = agent.invoke(
            {
                "query": query
            },
            config={
                "configurable": {
                    "thread_id": "streamlit-user"
                }
            }
        )

        answer = result.get(
            "final_answer",
            "Sorry, I could not generate a response."
        )

        answer = clean_answer(answer)

    except Exception as error:

        answer = (
            "Sorry, something went wrong while processing "
            "your request. Please try again."
        )

        print("Agent error:", error)


    
    # DISPLAY ASSISTANT RESPONSE
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)