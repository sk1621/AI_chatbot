import pickle

with open('final_chunk.pkl','rb') as f:
    final_chunks = pickle.load(f)

# print(final_chunks)

from langchain_community.embeddings import OllamaEmbeddings 
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

embeddings = OllamaEmbeddings(model="gemma:2b")

# r1 = embeddings.embed_documents(final_chunks)
# r1
docs = [Document(page_content=chunk) for chunk in final_chunks]
vectordb = Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="./chroma_db")
# print("Number of documents in vector DB:", vectordb._collection.count())
# query = "Salary"
# results = vectordb.similarity_search(query, k=2)  # fetch 2 matches
# for r in results:
#     print("\nMatched chunk:\n", r.page_content)

# res = vectordb.similarity_search(query="what will be the Salary for the YearsExperience between 1.5 to 3.0")
# for i in res:
#     print("\nMatched chunk:\n", i.page_content) ####### not showing the expected results 
####### should i store the datas in vectordb


