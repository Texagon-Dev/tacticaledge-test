# import os

# from langchain_community.llms import HuggingFaceTextGenInference

# ENDPOINT_URL = "https://qk1yb9fkb542cn50.us-east-1.aws.endpoints.huggingface.cloud"
# HF_TOKEN = "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"

# llm = HuggingFaceTextGenInference(
#     inference_server_url=ENDPOINT_URL,
#     max_new_tokens=512,
#     top_k=50,
#     temperature=0.1,
#     repetition_penalty=1.03,
#     server_kwargs={
#         "headers": {
#             "Authorization": f"Bearer {HF_TOKEN}",
#             "Content-Type": "application/json",
#         }
#     },
# )

# print(llm("hello"))

# If necessary, install the openai Python library by running 
# pip install openai

from openai import OpenAI

client = OpenAI(
	base_url="https://qk1yb9fkb542cn50.us-east-1.aws.endpoints.huggingface.cloud/v1/", 
	api_key="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"
)

chat_completion = client.chat.completions.create(
	model="tgi",
	messages=[
    {
        "role": "user",
        "content": "What is deep learning?"
    }
],
	stream=True,
	max_tokens=500
)

for message in chat_completion:
	print(message.choices[0].delta.content, end="")