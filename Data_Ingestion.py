from langchain_core.prompts import ChatPromptTemplate 
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = CSVLoader('salary_data.csv')
txt_loader = loader.load()
# print(txt_loader)

df = pd.read_csv('salary_data.csv')

chunk = []
for i in range(0, len(df),10):
    subset = df.iloc[i:i+10]
    chunk_file = subset.to_string(index=False)#header=(i==0)) another step.....
    chunk.append(chunk_file)

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

final_chunk=[]
for chunks in chunk:
    final_chunk.extend(splitter.split_text(chunks))

print(final_chunk)

import pickle

with open('final_chunk.pkl','wb') as f:
    pickle.dump(final_chunk,f)


