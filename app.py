from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
import pandas as pd
from langchain.schema import HumanMessage
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 


df = pd.read_csv('salary_data.csv')
df1 = pd.read_csv('random_data.csv')


llm = OllamaLLM(model='gemma:2b')

# embeddings = OllamaEmbeddings(model='gemma:2b')
# db = Chroma(persist_directory="./chroma_db",embedding_function=embeddings)
# retriever = db.as_retriever()
# print('num of docs in vectordb :',db._collection.count()) successfully loaded 

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You should act based on the mathematical operations i feeded.don't hallucinate"),
#     ("user","Context:\n{context}\n\nQuestion: {input}")
# ])

# doc_chain = create_stuff_documents_chain(llm,prompt)

# rag_chain = create_retrieval_chain(retriever,doc_chain)


# response = rag_chain.invoke({"input":"show the mean for the salary table in salary data file"})

st.title("Q&A chatbot")
st.write("ask your question here")
user_input = st.text_input("You :")

lst = []
def extract_col_name(question,df1):
    print(question)
    ques = question.lower()
    for col in df1.columns:
        if col.lower() in ques:
            lst.append(col)
    if lst:
        return lst
    else: 
        
        return 'please mention valid column names'
column = extract_col_name(user_input,df1)

def extract_operations(question,df1):
    ques = question.lower()
    lst = ['sum','mean','median','mode','describe'] #### problem for getting max and min..and also for mode(giving multiple values)
    matches = []  # local list
    for i in lst:
        if i in ques:
            matches.append((ques.index(i), i))
    if matches:
        matches.sort()                     
        return [m[1] for m in matches]    
    else:
        return 'none'
operation = extract_operations(user_input,df1)

def perform_operation(column,operation,df1):
    results = []
    for i in column:
        for j in operation:
            res = getattr(df1[i], j)()
            if isinstance(res, (int, float)):
                res = f"{res:,.2f}"  # adds commas and 2 decimal points
            results.append({j: res})
        return results
# result = perform_operation(column,operation,df)

# response = llm.invoke([HumanMessage(content=f"Question: {user_question}\nComputed results: {result}")])
if str(column).strip().lower() == 'please mention valid column names':
    prompt_text = f"""
    You are an AI assistant. 
    User Question: {user_input}
    Computed Results: please provide the proper input.
    """
    response = llm.invoke([HumanMessage(content=prompt_text)])
    st.write(response.content)

else:

    answer = perform_operation(column,operation,df1)
    print(answer)
    prompt_text = f"""
    You are an AI assistant. Here are the computed results for the user's question and also if the 
    user asks about to work on any charts asks the x and y column:

    User Question: {user_input}
    Computed Results: {answer}
    Please answer the question in a clear, human-readable, conversational format.dont hallucinate. show proper answer.
    """
    response = llm.invoke([HumanMessage(content=prompt_text)])
    
    st.write(response)

def extract_graph_word(question,df1):
    op=[]
    for col in df1.columns:
        if col.lower() in question:
            op.append(col)
            app = []
            lst = ['bar','pie','line','boxplot']
            ques = question.lower()
            for i in lst:
                if i in ques:
                    app.append(i)
    if col in op:
        for i in app:
            if i=='bar':
                fig = sns.barplot(data=df1,x=op[0],y=op[1])
            st.pyplot(fig)
            prompt_text = f"""
            You are an AI assistant.so its a visualization output.  
            User Question: {user_input}
            """
            response = llm.invoke([HumanMessage(content=prompt_text)])
            st.write(response.content)
    else:
        if col not in op:
            prompt_text = f"""
            You are an AI assistant. so this condition will tell us missing column names.
            User Question: {user_input}
            Computed Results: please provide the proper input.
            """
            response = llm.invoke([HumanMessage(content=prompt_text)])
            st.write(response.content)

    



