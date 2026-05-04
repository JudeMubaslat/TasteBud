# streamlit file
# pip install streamlit
import streamlit as st

from Step_3_V4 import initialize_messages, get_tastebud_response

# load the two images into the code
company_logo = "Images/Company Logo.png"
scout_icon = "Images/TasteBud Logo.png"

# spacing top
st.markdown("<br><br>", unsafe_allow_html=True)

# centered logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("Images/utensils.png", width=600)

st.markdown("""
    <div style='text-align: center;'>
        <h2 style='margin-bottom: 20px;'>TasteBud</h2>
        <p style='margin-top: 10px; font-size: 25px; color: #666;'>
            Your AI-powered recipe generator.
        </p>
    </div>
""", unsafe_allow_html=True)

# spacing
st.markdown("<br><br>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = initialize_messages()
# Display chat history (skip system message)
# this goes over the previous exchanges in the conversations and prints
# them in order.
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="👀").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant", avatar=scout_icon).write(msg["content"])
# Chat input
# allows the user to type in a new prompt
user_input = st.chat_input("What would you like to make?")

if user_input:
    # show user message
    st.chat_message("user", avatar="👀").write(user_input)

    # call model
    with st.spinner("TasteBud is thinking..."):
        response, updated_messages = get_tastebud_response(
            st.session_state.messages,
            user_input
        )

    # update memory
    st.session_state.messages = updated_messages

    # show assistant response
    st.chat_message("assistant", avatar=scout_icon).write(response)