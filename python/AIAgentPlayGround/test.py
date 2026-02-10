from langchain_ollama import ChatOllama

# 1. Setup the connection to your Unraid server
llm = ChatOllama(
    base_url="http://192.168.50.200:4567",  # Replace with your actual IP
    model="deepseek-r1"   # Replace with 'qwen2.5' or 'deepseek-r1:8b'
)

# 2. Send a simple message
response = llm.invoke("Hello! Are you ready to build an agent?")

# 3. Print the result
print(response.content)
