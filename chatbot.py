"""This module is designed to chat with LLAMA 2 using RAG and answer queries"""
#llama2
import os

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.llms import HuggingFaceTextGenInference


load_dotenv()


url= os.getenv("ENDPOINT_URL")
token = os.getenv("HF_TOKEN")



llm = HuggingFaceTextGenInference(
    inference_server_url=url,
    
    top_k=50,
    temperature=0.1,
    repetition_penalty=1.03,
    server_kwargs={
        "headers": {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
    },
)


GENERAL_SYSTEM_TEMPLATE_FOR_MOBILY = r""" 
               
  
                Role:
                You are a helpful chatbot assistant for "Mobily" company. 
                The question and context is for mthe Annual report 2022 of the company.
                Your job is to answer user questions the user accordingly in a conversational manner.

                question = {question}
                context = {context}

                output:
                give the most suitable answer.

                """


GENERAL_SYSTEM_TEMPLATE_FOR_OPERATIONS_MANUAL = r""" 
               
                Role:
                You are a helpful assistant for the Operation and Maintainance Manual. 
                The question and context you will get is from the Manual book.
                Your job is to asist the user accordinglt in a conversational manner.

                question = {question}
                context = {context}

                output:
                give the most suitable answer. 
                """


def chatting(user_input, db, input_id):
    """This is the function to chat with the model"""

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    general_user_template = "Question:```{question}```"

    if input_id == 1:
        messages = [
            SystemMessagePromptTemplate.from_template(
                GENERAL_SYSTEM_TEMPLATE_FOR_MOBILY
            ),
            HumanMessagePromptTemplate.from_template(general_user_template),
        ]
    elif input_id == 2:
        messages = [
            SystemMessagePromptTemplate.from_template(
                GENERAL_SYSTEM_TEMPLATE_FOR_OPERATIONS_MANUAL
            ),
            HumanMessagePromptTemplate.from_template(general_user_template),
        ]

    aqa_prompt = ChatPromptTemplate.from_messages(messages)
    retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": 5})
    conversation = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": aqa_prompt},
        verbose=False,
    )

    conversation.invoke({"question": user_input})
    # print(memory.buffer)
    return memory.buffer[-1].content
