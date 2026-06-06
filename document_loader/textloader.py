from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
data = TextLoader("document_loader/notes.txt")
docs = data.load()
template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant that summarizes the following text:"),
        ("human", "{data}")
    ]
)

model = ChatMistralAI(
    model="mistral-small-latest",   # or "mistral-large-latest"
    temperature=0.9,
    max_tokens=2048,
)
prompt = template.format_prompt(data=docs[0].page_content)
response=model.invoke(prompt)
print(response.content)

