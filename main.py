from dotenv import load_dotenv
import os
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size = 100,
    chunk_overlap=1
)

data = TextLoader("splitter/notes.txt")
docs = data.load()
model = ChatMistralAI(
    model="mistral-small-latest",   # or "mistral-large-latest"
    temperature=0.9,
    max_tokens=2048
)
chunks = splitter.split_documents(docs)
response = model.invoke(chunks[0].page_content)
print(response.content)