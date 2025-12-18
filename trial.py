import pandas as pd
from sqlalchemy import create_engine 
import streamlit as st
from langchain_ollama import OllamaLLM
# from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import seaborn as sns 
import matplotlib as plt 
from decimal import Decimal
import pyarrow as pa
import plotly.express as pxs
import sqlite3
import sqlalchemy
from dotenv import load_dotenv
load_dotenv()
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os 

os.environ['NVIDIA_API'] = os.getenv('NVIDIA_api_key')

client = ChatNVIDIA(model="qwen/qwen2.5-coder-32b-instruct",
  api_key=os.environ['NVIDIA_API'], 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,)

response = client.invoke([{'role':'user','content':'whats the concept of multi agents'}])
print(response.content)