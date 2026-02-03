from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine


def procesar_datos_apple(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    return df

def filtrar_por_fecha(df, fecha_inicio, fecha_fin):
    return df[df["Date"].between(fecha_inicio, fecha_fin)]

