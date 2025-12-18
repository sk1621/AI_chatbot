import pandas as pd
from sqlalchemy import create_engine 
import streamlit as st
import plotly.express as px
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
# import seaborn as sns 
# import matplotlib as plt 
# from decimal import Decimal
# import pyarrow as pa
# from langchain_ollama import OllamaLLM
# from langchain_core.messages import HumanMessage
# import sqlite3
# import sqlalchemy
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
load_dotenv()

os.environ['api_key_groq']= os.getenv('Groq_api_key')
os.environ['api_key_huggingface'] = os.getenv('Hugging_face_api')

llm = ChatGroq(groq_api_key = os.environ['api_key_groq'],model="openai/gpt-oss-120b")
# llm = OllamaLLM(model="gemma:2b")
# os.environ['NVIDIA_API'] = os.getenv('NVIDIA_api_key')

# llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct",
#   api_key=os.environ['NVIDIA_API'], 
#   temperature=0.2,
#   top_p=0.7,
#   max_completion_tokens=1024,)

st.title('Q&A Chatbot')

if "show_db_form" not in st.session_state:
    st.session_state.show_db_form = False
if "table_list" not in st.session_state:
    st.session_state.table_list = None
if "table_name" not in st.session_state:
    st.session_state.table_name = None
if "table_name_second" not in st.session_state:
    st.session_state.table_name_second = None
if "engine" not in st.session_state:
    st.session_state.engine = None
if "df" not in st.session_state:
    st.session_state.df = None
if "df_1" not in st.session_state:
    st.session_state.df_1 = None
if "connect_clicked" not in st.session_state:
    st.session_state.connect_clicked = None

if st.button("➕ DB Connect"):
    st.session_state.show_db_form = True
    st.info("Fill MySQL Credentials on your left")

if st.session_state.show_db_form:
    user = st.sidebar.text_input("Username", value="root")
    password = st.sidebar.text_input("Password", value="root")
    host = st.sidebar.text_input("Host", value="localhost")
    port = st.sidebar.number_input("Port", value=3306, step=1)
    database = st.sidebar.text_input("Database name",value='new_db')
    st.session_state.connect_clicked = st.sidebar.button("Connect")

if st.session_state.connect_clicked and st.session_state.engine is None and database.strip():
    try:
        st.session_state.engine = create_engine(
        f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        )

        tables = pd.read_sql("SHOW TABLES;", con=st.session_state.engine)
        st.session_state.table_list = tables.iloc[:, 0].tolist()

        st.success(f"Connected to database: {database}")

    except Exception as e:
        st.error(f"Connection failed: {e}")

table_name = st.session_state.table_name
table_name_second = st.session_state.table_name_second



def extract_assistant_text(engine, table_list,tables=None):
    if engine and table_list:
            table_name = st.selectbox("Select a table", table_list, key="table1")
            table_name_second = st.selectbox("Select a second table (optional)", table_list, key="table2")
            if tables is None:
                tables = []
            if table_name:
                tables.append(table_name)
                tables_sql = ', '.join([f"'{t}'"for t in tables])
                schema_sql = f"""
                        SELECT 
                            table_name,
                            column_name,
                            data_type
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE table_schema = DATABASE()
                        AND table_name IN ({tables_sql});
                    """
                df_schema = pd.read_sql(schema_sql, con=engine)
                st.session_state.df_schema = df_schema

            if "primary_table_loaded" not in st.session_state:
                st.session_state.primary_table_loaded = False
            if "both_table_loaded" not in st.session_state:
                st.session_state.both_table_loaded = False

            load_primary = st.button("Load Primary Table", key="btn_load_primary")
            load_both = st.button("Load Both Tables", key="btn_load_both")

            if load_primary:
                st.session_state.df = pd.read_sql(
                    f"SELECT * FROM {table_name}",
                    con=engine
                )
                st.success(f"Loaded table: {table_name}")
                st.session_state.primary_table_loaded = True
                st.session_state.both_table_loaded = False

            if st.session_state.primary_table_loaded:
                st.write(st.session_state.df.head(5),st.session_state.df.shape,'shows the no of rows and columns in this table')

            db = SQLDatabase(engine=engine, include_tables=[table_name])
            sql_tool = QuerySQLDataBaseTool(db=db)
            agent = create_agent(llm, tools=[sql_tool])
            st.session_state.agent = agent
        
            if load_both:
                if table_name and table_name_second:
                    tables.append(table_name_second)
                    tables_sql = ', '.join([f"'{t}'"for t in tables])
                    schema_sql = f"""
                            SELECT 
                                table_name,
                                column_name,
                                data_type
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE table_schema = DATABASE()
                            AND table_name IN ({tables_sql});
                        """
                    df_schema_multi = pd.read_sql(schema_sql, con=engine)
                    st.session_state.df_schema_multi = df_schema_multi
                if table_name == table_name_second:
                    st.error("Please select two different tables.")
                else:
                    st.session_state.df = pd.read_sql(
                        f"SELECT * FROM {table_name}",
                        con=engine
                    )
                    st.session_state.df_1 = pd.read_sql(
                        f"SELECT * FROM {table_name_second}",
                        con=engine
                    )

                    st.success(f"Loaded table: {table_name}")
                    st.success(f"Loaded table: {table_name_second}")

                    st.session_state.primary_table_loaded = False
                    st.session_state.both_table_loaded = True

            if st.session_state.both_table_loaded:
                st.write(st.session_state.df.head(5),st.session_state.df.shape,'shows the no of rows and columns in this table')
                st.write(st.session_state.df_1.head(5),st.session_state.df.shape,'shows the no of rows and columns in this table')

            both_table = SQLDatabase(engine=engine,
                            include_tables=[table_name, table_name_second])
            sql_tool_both = QuerySQLDataBaseTool(db=both_table)
            agent = create_agent(llm, tools=[sql_tool_both])
            agent_tables = create_agent(llm, tools=[sql_tool_both])
            st.session_state.agents = agent_tables 

    df_schema = st.session_state.get("df_schema")
    df_schema_multi = st.session_state.get("df_schema_multi")
    agent = st.session_state.get("agent")
    agent_tables = st.session_state.get("agents")

    st.write("Ask Something here")
    user_input =  st.text_input("You :")
   
    if user_input:
            prompt = f"""
            You are a SQL assistant. Convert the user's question into SQL, execute it, and show the result.
            NEVER perform a JOIN unless the join keys actually exist in BOTH tables.
            1.Before generating SQL, validate that:
                - The table names exist
                - The join columns exist in both tables
            2. If a requested join column does NOT exist:
                → Show invalid join columns.select proper tables for executing join query.
            User Input: {user_input}
            If the values are in float round off it to two decimals.
            Do Not show the sql query.
            If result_text not in session_state → call agent
            Else → DO NOT call agent again
            Do Not explain the queries.
            show the results in table.
            """
            response = None
            if st.button and agent != None:
                response = agent.invoke({
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            elif st.button and agent_tables != None:
                response = agent_tables.invoke({
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            else:
                st.write("Agents not passing")

            result_text=""
            if "result_text" not in st.session_state:
                st.session_state.result_text = None

            if isinstance(response, dict) and "messages" in response:
                msgs = response["messages"]

            elif hasattr(response, "messages"):
                msgs = response.messages

            else:
                msgs = None
                result_text = str(response)

            if msgs:
                last = msgs[-1]

                if isinstance(last, dict):
                    result_text = (
                        last.get("content")
                        or last.get("text")
                        or str(last)
                    )
                else:
                    if hasattr(last, "content"):
                        result_text = last.content
                    elif hasattr(last, "text"):
                        result_text = last.text
                    else:
                        result_text = str(last)

            if result_text.strip() != "None":
                    st.session_state.result_text = result_text
            else:
                st.write('The error might be wrong user question')

            if st.session_state.result_text:
                st.write(st.session_state.result_text)
                
            content = st.session_state.result_text
            print("successfully content showing")

            ######## session_state for df and download button for excel and csv file
            def markdown_table(markdown_text):
                    if "data_out" not in st.session_state:
                        st.session_state.data_out = None
                    lines = [line.strip() for line in markdown_text.split("\n") if line.strip().startswith("|")]
                    if len(lines)< 2:
                        None
                    header = [col.strip() for col in lines[0].strip("|").split("|")]
                    data_rows = []

                    for row in lines[2:]:
                        row_line =  [cols.strip() for cols in row.strip("|").split("|")]
                        data_rows.append(row_line)
                        df = pd.DataFrame(data_rows, columns=header)
                    # st.write(df.dtypes)
                        for col in df.columns:
                            col_lower = col.lower()
                            if "year" in col_lower or "years" in col_lower:
                                df[col] = df[col].astype(str)

                            elif "month" in col_lower or "months" in col_lower:
                                df[col] = df[col].astype(str)
                        for col in df.columns:
                            if "year" in col.lower() or "month" in col.lower() or "years" in col.lower() or "months" in col.lower():
                                continue
                            cleaned = df[col].astype(str).str.replace(",", "").str.strip()
                            df[col] = pd.to_numeric(cleaned, errors="ignore")

                        st.session_state.data_out = df

                    return df
            
            df = markdown_table(content)
            
            # if df is not None:
            #     st.write("Data types:", df.dtypes)
            # else:
            #     st.error("No table found in output.")

            numeric_cols = df.select_dtypes(exclude='object').columns.tolist()
            
            categorical_cols = df.select_dtypes(include='object').columns.tolist()

            datetime_cols = [
                col for col in df.columns
                if "date" in col.lower() or "time" in col.lower() or "month" in col.lower() or "years" in col.lower()
            ]

            valid_charts = []

            if categorical_cols and numeric_cols:
                valid_charts += ["Bar Chart", "Column Chart", "Pie Chart", "Donut Chart", "Pareto Chart"]

            if len(categorical_cols) >= 2:
                valid_charts += ["Grouped Bar Chart", "Stacked Column Chart"]

            if datetime_cols and numeric_cols:
                valid_charts += ["Line Chart", "Area Chart"]

            st.session_state.setdefault("chart_type", None)
            st.session_state.setdefault("x_col", None)
            st.session_state.setdefault("y_col", None)
            st.session_state.setdefault("x_col_2",None)

            chart_type = st.selectbox(
                "Select chart type",
                ["Select one"] + valid_charts,
                key="chart_type_sel"
            )

            if chart_type != "Select one":
                st.session_state.chart_type = chart_type

            if not st.session_state.chart_type:
                st.stop()

            chart_type = st.session_state.chart_type
            if chart_type in ["Bar Chart", "Column Chart", "Pie Chart", "Donut Chart", "Pareto Chart"]:
                st.session_state.x_col = st.selectbox("Categorical Column", categorical_cols, key="xx1")
                st.session_state.y_col = st.selectbox("Numeric Column", numeric_cols, key="yy1")

            elif chart_type in ["Line Chart", "Area Chart"]:
                st.session_state.x_col = st.selectbox("Date Column", datetime_cols, key="xx2")
                st.session_state.y_col = st.selectbox("Numeric Column", numeric_cols, key="yy2")

            elif chart_type in ["Grouped Bar Chart", "Stacked Column Chart"]:
                st.session_state.x_col = st.selectbox("Category 1", categorical_cols, key="xx3")
                st.session_state.x_col_2 = st.selectbox(
                    "Category 2",
                    [c for c in categorical_cols if c != st.session_state.x_col],
                    key="yy3"
                )
                st.session_state.y_col = st.selectbox("Numeric Column",numeric_cols,key="yy_1")

            if not st.session_state.x_col or not st.session_state.y_col:
                st.stop()

            x = st.session_state.x_col
            y = st.session_state.y_col
            x_col = st.session_state.x_col_2
            st.subheader(chart_type)

            if chart_type == "Bar Chart":
                st.bar_chart(df.set_index(x)[y])

            elif chart_type == "Column Chart":
                st.plotly_chart(px.bar(df, x=x, y=y))

            elif chart_type == "Pie Chart":
                st.plotly_chart(px.pie(df, names=x, values=y))

            elif chart_type == "Donut Chart":
                st.plotly_chart(px.pie(df, names=x, values=y, hole=0.4))

            elif chart_type == "Line Chart":
                st.plotly_chart(px.line(df, x=x, y=y))

            elif chart_type == "Area Chart":
                st.plotly_chart(px.area(df, x=x, y=y))

            elif chart_type == "Grouped Bar Chart":
                st.plotly_chart(px.bar(df, x=x,y=y, color=x_col))

            elif chart_type == "Stacked Column Chart":
                fig = px.bar(df, x=x, color=x_col,y=y)
                fig.update_layout(barmode="stack")
                st.plotly_chart(fig)
            else:
                st.write("U might selected the wrong type of chart or unmatched chart type")
            print("successfully chart showing")

extract_assistant_text(st.session_state["engine"],st.session_state["table_list"])
            


########### monitor groq tokens and implement langsmith monitoring 
############ still having the problems like again and again running the user prompt while selecting the chart type
########## will this rectify using multiple agents 
############# Nvidia NIM not satsfied with the results, since it fails in accuracy 

