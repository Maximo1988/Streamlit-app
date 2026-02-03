import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import db_connect

engine = db_connect()

st.header('datos de pruebas')
st.subheader('pruebitas con fechas')
df = pd.read_csv('https://raw.githubusercontent.com/it-ces/Datasets/refs/heads/main/AAPL.csv')
st.dataframe(df.tail())

df['Date'] = pd.to_datetime(df['Date'])

fecha_inicio, fecha_fin = st.date_input("Rango de fechas",value=[datetime.date(2024, 1, 1), datetime.date.today()])

fecha_inicio = pd.Timestamp(fecha_inicio)
fecha_fin = pd.Timestamp(fecha_fin)

df_rango = df[df["Date"].between(fecha_inicio, fecha_fin)]
st.dataframe(df_rango)

hist = px.histogram(df_rango['Open'])
st.plotly_chart(hist)

df_rango['month'] = df_rango['Date'].dt.month

box = px.box(df_rango, x='month', y='Close')
st.plotly_chart(box)