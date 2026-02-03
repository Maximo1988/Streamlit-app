import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import procesar_datos_apple, filtrar_por_fecha, calcular_retornos, analizar_convergencia_precios, kmeans_clustering, random_forest_prediccion

st.set_page_config(page_title="Apple Stock Analysis", layout="wide")
st.title("📈 Análisis de Acciones de Apple 23/24")

url = 'https://raw.githubusercontent.com/it-ces/Datasets/refs/heads/main/AAPL.csv'
df = pd.read_csv(url)
df = procesar_datos_apple(df)

st.sidebar.header("Filtro")
fecha_inicio, fecha_fin = st.sidebar.date_input("Selecciona el Rango de fechas", value=[datetime.date(2024, 1, 1), datetime.date.today()], max_value=datetime.date.today())
fecha_inicio = pd.Timestamp(fecha_inicio)
fecha_fin = pd.Timestamp(fecha_fin)
df_filtrado = filtrar_por_fecha(df, fecha_inicio, fecha_fin)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Convergencia", "💰 Retornos", "🎯 K-Means", "🤖 Random Forest"])

with tab1:
    st.subheader("Análisis de Convergencia y Tendencias")
    with st.expander("ℹ️", expanded=False):
        st.write("Muestra 3 líneas:\n1. **Precio actual** (línea azul fina)\n2. **Promedio de 30 días** (línea naranja) - tendencia reciente\n3. **Promedio de 90 días** (línea verde) - tendencia general\n\nSi el precio está arriba = Está subiendo 📈\nSi el precio está abajo = Está bajando 📉")
    
    df_convergencia = analizar_convergencia_precios(df_filtrado)
    fig_convergencia = go.Figure()
    fig_convergencia.add_trace(go.Scatter(x=df_convergencia['Date'], y=df_convergencia['Close'], name='Precio', line=dict(color='#1f77b4', width=1)))
    fig_convergencia.add_trace(go.Scatter(x=df_convergencia['Date'], y=df_convergencia['MA_30'], name='MA 30d', line=dict(color='#ff7f0e', width=2)))
    fig_convergencia.add_trace(go.Scatter(x=df_convergencia['Date'], y=df_convergencia['MA_90'], name='MA 90d', line=dict(color='#2ca02c', width=2)))
    fig_convergencia.update_layout(title='Convergencia de Precios', xaxis_title='Fecha', yaxis_title='Precio (USD)', height=500)
    st.plotly_chart(fig_convergencia, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    precio_actual = df_convergencia['Close'].iloc[-1]
    ma30 = df_convergencia['MA_30'].iloc[-1]
    ma90 = df_convergencia['MA_90'].iloc[-1]
    std30 = df_convergencia['std_30'].iloc[-1]
    with col1:
        st.metric("Precio Actual", f"${precio_actual:.2f}")
    with col2:
        st.metric("MA 30d", f"${ma30:.2f}")
    with col3:
        st.metric("MA 90d", f"${ma90:.2f}")
    with col4:
        st.metric("Desv Est 30d", f"${std30:.2f}")

with tab2:
    st.subheader("Análisis de Retornos")
    with st.expander("ℹ️", expanded=False):
        st.write("Muestra cuánto dinero ganaste o perdiste:\n- **Retorno Total**: Ganancia/pérdida desde el inicio (%)\n- **Retorno Diario**: Cambio promedio cada día\n- **Volatilidad**: Qué tan inestable es (número alto = más cambios)\n- **Máx Retorno**: Mejor ganancia en 1 día")
    
    df_retornos = calcular_retornos(df_filtrado)
    fig_retornos = px.line(df_retornos, x='Date', y='retorno_acumulado', title='Retorno Acumulado (%)', height=400)
    st.plotly_chart(fig_retornos, use_container_width=True)
    
    fig_hist = px.histogram(df_retornos.dropna(), x='retorno_diario', nbins=50, title='Distribución Retornos Diarios', height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    ret_total = df_retornos['retorno_acumulado'].iloc[-1]
    ret_diario = df_retornos['retorno_diario'].mean()
    vol = df_retornos['retorno_diario'].std()
    max_ret = df_retornos['retorno_diario'].max()
    with col1:
        st.metric("Retorno Total", f"{ret_total:.2f}%")
    with col2:
        st.metric("Retorno Diario Prom", f"{ret_diario:.3f}%")
    with col3:
        st.metric("Volatilidad", f"{vol:.2f}%")
    with col4:
        st.metric("Máx Retorno", f"{max_ret:.2f}%")

with tab3:
    st.subheader("K-Means Clustering")
    with st.expander("ℹ️", expanded=False):
        st.write("Agrupa los días en 2-5 grupos según su precio y volumen.\n\nCada color = grupo diferente\nTe muestra qué días fueron parecidos (mismo comportamiento).")
    
    n_clusters = st.slider("Número de grupos", 2, 5, 3)
    try:
        df_clusters, kmeans_model = kmeans_clustering(df_filtrado, n_clusters=n_clusters)
        fig_clusters = px.scatter(df_clusters, x='Date', y='Close', color='cluster', title=f'Clustering (K={n_clusters})', height=500)
        st.plotly_chart(fig_clusters, use_container_width=True)
        cluster_stats = df_clusters.groupby('cluster').agg({'Close': ['min', 'mean', 'max'], 'Volume': 'mean'}).round(2)
        st.dataframe(cluster_stats, use_container_width=True)
        for cid in range(n_clusters):
            cdata = df_clusters[df_clusters['cluster'] == cid]
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Grupo {cid} - Precio Medio", f"${cdata['Close'].mean():.2f}")
            with col2:
                st.metric(f"Grupo {cid} - Volumen", f"{cdata['Volume'].mean():,.0f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

with tab4:
    st.subheader("Predicción Random Forest")
    with st.expander("ℹ️", expanded=False):
        st.write("Predice si el precio subirá o bajará mañana.\n\n- **Exactitud**: % de veces que acertó\n- **Predicción**: SUBIDA o BAJADA\n- **Confianza**: Seguridad de la predicción\n\n⚠️ NO es 100% preciso, es solo un análisis técnico.")
    
    try:
        resultado = random_forest_prediccion(df_filtrado)
        if resultado[0] is None:
            st.warning("No hay datos suficientes")
        else:
            rf, acc, feat_imp, pred, prob, historico = resultado
            
            # Gráfica GAUGE:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_gauge_acc = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=acc*100,
                    title={'text': "Exactitud del Modelo (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffcccc"},
                            {'range': [50, 75], 'color': "#ffffcc"},
                            {'range': [75, 100], 'color': "#ccffcc"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge_acc.update_layout(height=300)
                st.plotly_chart(fig_gauge_acc, use_container_width=True)
            
            with col2:
                fig_gauge_conf = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob[pred]*100,
                    title={'text': "Confianza de Predicción (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ca02c" if pred == 1 else "#d62728"},
                        'steps': [
                            {'range': [0, 60], 'color': "#ffcccc"},
                            {'range': [60, 80], 'color': "#ffffcc"},
                            {'range': [80, 100], 'color': "#ccffcc"}
                        ]
                    }
                ))
                fig_gauge_conf.update_layout(height=300)
                st.plotly_chart(fig_gauge_conf, use_container_width=True)
            
            # Aviso de Card grande:
            pred_txt = "📈 SUBIDA" if pred == 1 else "📉 BAJADA"
            pred_color = "#d4edda" if pred == 1 else "#f8d7da"
            st.markdown(f"""
            <div style="background-color: {pred_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="margin: 0; font-size: 1.2em;">Predicción para Mañana</h2>
                <h1 style="margin: 10px 0; font-size: 1.5em;">{pred_txt}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            
            # Grafica de Histórico Visual
            st.subheader("📊 Histórico: Predicciones vs Realidad (Últimos 60 días)")
            
            fig_historico = go.Figure()
            
            # Realidad
            fig_historico.add_trace(go.Scatter(
                x=historico['Date'],
                y=historico['target'],
                mode='lines+markers',
                name='Realidad (0=Bajó, 1=Subió)',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            
            # Predicción
            fig_historico.add_trace(go.Scatter(
                x=historico['Date'],
                y=historico['prediccion'],
                mode='lines+markers',
                name='Predicción del Modelo',
                line=dict(color='#ff7f0e', width=2, dash='dot'),
                marker=dict(size=6, symbol='x')
            ))
            
            fig_historico.update_layout(
                title='Comparación: Predicción vs Realidad',
                xaxis_title='Fecha',
                yaxis_title='Dirección (0=Bajó, 1=Subió)',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_historico, use_container_width=True)
            
            # Calcular aciertos
            aciertos = (historico['target'] == historico['prediccion']).sum()
            total = len(historico)
            tasa_acierto = (aciertos / total) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Días Analizados", total)
            with col2:
                st.metric("Aciertos", aciertos)
            with col3:
                st.metric("Tasa de Acierto", f"{tasa_acierto:.1f}%")
            
            st.write("---")
            
            # Feature importance
            st.subheader("📈 Factores Más Importantes")
            fig_imp = px.bar(feat_imp, x='importance', y='feature', orientation='h', title='Importancia de Características', labels={'importance': 'Importancia', 'feature': 'Características'}, height=400)
            st.plotly_chart(fig_imp, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

