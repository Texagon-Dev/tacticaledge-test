"""Creating The streamlit app"""
import streamlit as st
from chatbot import chatting
from embedd_docs import get_db


def doc1():
    """Module to create embeddings for doc1"""
    db = get_db("mobily")
    st.write("This is the Mobily Document")
    query = st.text_input("Enter your Question")
    if st.button("Submit"):
        st.write(chatting(query, db, 1))


def doc2():
    """Module to generate embeddings for doc2"""
    db = get_db("operation")
    st.write("This is the Operations And Maintainance Document")
    query = st.text_input("Enter your Question")
    if st.button("Submit"):
        st.write(chatting(query, db, 2))


menu = ["Mobily Document", "Operations And Maintainance Document"]
choice = st.sidebar.selectbox("Select Document to Chat", menu)


if choice == "Mobily Document":
    doc1()
elif choice == "Operations And Maintainance Document":
    doc2()
