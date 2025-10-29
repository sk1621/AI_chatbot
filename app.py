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
df1 = pd.read_csv('D:/Data/Amazon Sale Report.csv')


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

def extract_col_name(question, df1):
    print(question)
    ques = question.lower()
    if 'bar' not in ques:
        for col in df1.columns:
            if col.lower() in ques:
                lst.append(col)
        if lst:
            return lst
        else:
            return  
    else: 
        return
column = extract_col_name(user_input,df1)
print(column)

lst_1 = ['sum','mean','median','mode','describe'] 
matches = [] 

def extract_operations(question,df1):
    ques = question.lower()
    #### problem for getting max and min..and also for mode(giving multiple values)
     # local list
    for i in lst_1:
        if i in ques:
            matches.append((ques.index(i), i))
    if matches:
        matches.sort()                     
        return [m[1] for m in matches]    
    # else:
    #     return 'none'
operation = extract_operations(user_input,df1)
print(operation)

results = []
def perform_operation(column,operation,df1):
    for i in column:
        for j in operation:
            if j in lst_1:
                res = getattr(df1[i], j)()
                if isinstance(res, (int, float)):
                    res = f"{res:,.2f}"  # adds commas and 2 decimal points
                results.append({j: res})
        return results
# result = perform_operation(column,operation,df)

# response = llm.invoke([HumanMessage(content=f"Question: {user_question}\nComputed results: {result}")])

if column == lst:
    if operation is None:
        st.write("No operation provided. Please give a valid operation.")  
    else:
        answer = perform_operation(column, operation, df1)
        prompt_text = f"""
        You are an AI assistant. Here are the computed results for the user's question.
        User Question: {user_input}
        Computed Results: {answer}
        Please answer the question in a clear, human-readable, conversational format.
        """
        response = llm.invoke(prompt_text)
        st.write(response)

# else:
#     prompt_text = f""" 
#     User Question: {user_input}
#     Computed Results: please provide the proper input.
#     """
#     response = llm.invoke(prompt_text)
#     st.write(response)

app = []
op = []

def extract_graph_word(question, df1):
    print(question)
    question = question.lower()
    chart_keywords = ['bar','pie','line','boxplot']

    if not any(keyword in question for keyword in chart_keywords):
        print("No chart keyword found. Exiting function.")
        return st.write("Not mentioning the specific chart. pls mention chart names")  
    
    for col in df1.columns:
        if col.lower() in question:
            op.append(col)
        
        if len(op) == 2:
            for i in chart_keywords:
                if i in question:
                    app.append(i)
            
            for i in app:
                if i == 'bar':
                    print('yes')
                    fig, ax = plt.subplots()
                    sns.barplot(data=df1, x=op[0], y=op[1], ax=ax)
                    ax.set_title(f"Bar plot of {op[0]} vs {op[1]}")

                    prompt_text = f"""
                    You are an AI assistant. DO NOT show the example, definition of bar charts.
                    just show only the answer.
                    User Question: {user_input}
                    """
                    response = llm.invoke(prompt_text)
                    st.write(response.content if hasattr(response, "content") else response)
                    st.pyplot(fig)
                    plt.close(fig)   ####### should add the explanation about the chart... try openai....               
                    return              ########## try using plotly charts & most importantly chart should be done based                         
                else:                   # on conditions. we cant visualize bar chart for randomly passing columns...!!!!
                    print('no')
                    return
    if len(op) != 2:
        prompt_text = f"""
        You are an AI assistant.
        User Question: {user_input}
        """
        response = llm.invoke(prompt_text)
        st.write(response.content if hasattr(response, "content") else response)


if user_input:
    extract_graph_word(user_input, df1)

    



