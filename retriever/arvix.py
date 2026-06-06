from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True,
)

docs = retriever.invoke("deep learning")

for doc in docs:
    print(f"Title: {doc.metadata['Title']}")
    print(f"Authors: {', '.join(doc.metadata['Authors'])}")
    print(f"Abstract: {doc.page_content}")
    print(f"Published: {doc.metadata['Published']}")
    print(f"URL: {doc.metadata['entry_id']}")
    print("\n---\n")