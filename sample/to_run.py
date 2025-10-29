import streamlit as st
import matplotlib.pyplot as plt
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

llm = OllamaLLM(model='gemma:2b')
# df1 = pd.read_csv('D:/filtered_data/overall.csv')

st.title("Q&A chatbot")
st.write("ask your question here")
user_input = st.text_input("You :")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
# Step 2: Read and store file
if uploaded_file is not None:
    
    if uploaded_file.name.endswith('.csv'):
        df1 = pd.read_csv(uploaded_file)
    else:
        df1 = pd.read_excel(uploaded_file)
    st.success("File uploaded and loaded successfully!")
    st.write("Preview of data:")
    st.dataframe(df1.head(2))
else:
    st.warning("Please upload a CSV or Excel file to continue.")

# submit = st.button('submit')
###### if user asks about summary this should work and graph ideas another prompt should work....
question = ['graph', 'plot', 'visualization', 'chart','visual']
summary_keywords = [
    "summary", "summarize", "overview", "describe", "explain data", 
    "data info", "dataset details", "basic stats", "data summary", "insight", "findings", "key points", "pattern", "highlight",
    "data understanding", "EDA", "exploratory analysis"]
@st.cache_data
def summary(user_input, df1):
    summary_text = f""" Dataset has {df1.shape[0]} rows and {df1.shape[1]} columns.
    Numeric columns: {df1.select_dtypes(exclude='object').columns.tolist()},
    Categorical columns: {df1.select_dtypes(include='object').columns.tolist()},
    Correlations: {df1.select_dtypes(exclude='object').corr()},
    Column names: {df1.columns.tolist()}. Data types: {df1.dtypes.to_dict()},
    Missing values by column: {df1.isnull().sum().to_dict()},
    Data description: {df1.describe(include='all')} """ # here need to work on decimals.......
    
    prompt_text = f"""
    You are a precise data analysis assistant.

    Below is a structured summary of a dataset, including row/column counts,
    column types, correlations, and descriptive statistics.

    Dataset Summary:
    {summary_text}

    User Question: {user_input}

    Task:
    - Use ONLY the information available in the dataset summary.
    - Do NOT create or assume values not mentioned in the summary.
    - work on all the summary DO NOT skip any steps.
    """
    print('showing summary result')
    response = llm.invoke(prompt_text)
    st.write(response)

@st.cache_data
def graph_ideas(df1,user_input):
    numeric_cols = df1.select_dtypes(exclude='object').columns.tolist()
    print(numeric_cols)
    categorical_cols = df1.select_dtypes(include='object').columns.tolist()
    print(categorical_cols)
    datetime_cols = [col for col in df1.columns if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower()]
    
    prompt_text = f"""
            You are a data visualization assistant. 
            Your goal is to analyze the dataset summary and suggest all possible chart types that can be created using the available columns.

            summary_info = {{
            "Numeric columns": {numeric_cols},
            "Categorical columns": {categorical_cols},
            "Datetime columns": {datetime_cols}
            }}

            User Question: {user_input}
            
            ### Rules for chart suggestions:
            1. Use only the given columns from summary_info. Do NOT create your own examples.
            2. Suggest **all possible chart combinations**, not just one.
            3. Follow these mapping rules strictly:

            #### Core Comparison & Composition Charts
            - If one categorical and one numeric column → suggest "Bar Chart", "Column Chart", "Grouped Bar Chart", "Stacked Column Chart", "Pareto Chart", "KPI Indicator Cards".
            - If one categorical and one numeric (small number of categories ≤ 6) → also suggest "Pie Chart" and "Donut Chart".
            - If two categorical columns → suggest "Grouped Bar Chart" or "Stacked Column Chart".

            #### Correlation & Trend Charts
            - If two numeric columns → suggest "Scatter Plot" and "Bubble Chart".
            - If one datetime and one numeric column → suggest "Line Chart" or "Area Chart".
            - If one datetime and one categorical column → suggest "Grouped Bar Chart" (by time).

            #### Distribution & Relationship Charts
            - If one numeric column → suggest "Histogram" or "Box Plot".
            - If multiple numeric columns → suggest "Heatmap" (correlation matrix).

            ### Output Rules (STRICT):
            - Only show chart types that match the columns logically.
            - List **all valid chart combinations** based on the above rules.
            - Format each line exactly like this:
              Chart Type: <chart_name>, X: <column_name>, Y: <column_name>
            - Do not explain or describe anything.
            - If a chart type is not applicable (e.g., only one categorical column for stacked bar), skip it.
            - Do not invent or rename columns.
            """
    print('successfully showing graph ideas')
    response = llm.invoke(prompt_text)
    st.write(response)

user_input = user_input.lower()

words = word_tokenize(user_input)
snowball = SnowballStemmer('english')

if user_input:

    if any(word in user_input.lower() for word in summary_keywords) and any(word for word in question if any(snowball.stem(word) in snowball.stem(w) for w in words)):
        summary(user_input, df1)
        graph_ideas(df1,user_input)

    elif any(word in user_input.lower() for word in summary_keywords):
        summary(user_input, df1)

    elif any(word for word in question if any(snowball.stem(word) in snowball.stem(w) for w in words)):
        graph_ideas(df1,user_input)

    else:
        st.error("Couldn't identify your intent.")


##### check the report passing files also....
#### api integration part
#### file uploading from front end using streamlit.....
##### use class functions in this file to pass the function to app.py....
##### how to differentiate these two...????
###### after uploading the file. it should suggest two buttons one is summary another one is graph...