# streamlit file
# pip install streamlit
import streamlit as st
from Step_3_V4 import initialize_messages, get_tastebud_response

# load the two images into the code
company_logo = "Images/Company Logo.png"
scout_icon = "Images/TasteBud Logo.png"

# this sets up the name in the browser tab
st.set_page_config(
page_title="TasteBud",
layout="centered"
)
# add the company logo on top of the page
st.image(company_logo, width = 600)
# add a title under the company logo
st.title("TasteBud – Recipe Generator")
# Initialize conversation memory once per session
# the conversation is initialized with the system prompt
# this code is using function initialize_messages() in the other file
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

# replace the session_state.messages with the updated list of messages
#st.session_state.messages = updated_messages
# display the LLMs latest response
#st.chat_message("assistant", avatar=scout_icon).write(response)