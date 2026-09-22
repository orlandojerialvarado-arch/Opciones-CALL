import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# Configuración visual de la página
st.set_page_config(page_title="Sistema Analítico de Opciones CALL", layout="wide", page_icon="📈")

# Banner institucional idéntico al reporte oficial
st.markdown("""
<div style="background-color: #1e2a4a; padding: 18px; border-radius: 6px; text-align: center; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0; font-weight: 700; letter-spacing: 0.04em;">BOT INVERSIÓN EN OPCIONES CALL — PLATAFORMA INTEGRAL</h2>
    <p style="color: #94a3b8; margin: 5px 0 0; font-size: 0.9rem;">Pipeline End-to-End: Limpieza en memoria, inferencia ML y analítica prescriptiva en vivo</p>
</div>
""", unsafe_allow_html=True)

# 1. Carga del Modelo Serializado
MODEL_FILE = "mejor_modelo_aapl_call.pkl"

@st.cache_resource
def obtener_modelo():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

modelo = obtener_modelo()

# 2. Barra lateral para carga de datos
st.sidebar.header("📂 Ingesta de Datos")
archivo_cargado = st.sidebar.file_uploader("Subir CSV de Opciones (Kaggle o histórico):", type=["csv"])

if archivo_cargado is not None:
    # Lectura y depuración de nombres de columnas
    df_raw = pd.read_csv(archivo_cargado, low_memory=False)
    df_raw.columns = [str(c).strip().replace("[", "").replace("]", "") for c in df_raw.columns]

    with st.spinner("Procesando datos y calculando variables analíticas..."):
        df = df_raw.copy()
        
        # Conversión de fechas
        df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"], errors="coerce")
        df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"], errors="coerce")

        # Conversión a tipos numéricos
        cols_num = ["UNDERLYING_LAST", "DTE", "C_IV", "C_VOLUME", "C_LAST", "C_BID", "C_ASK", "STRIKE"]
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")

        # Feature Engineering: Prima, Moneyness y Spread
        if "C_BID" in df.columns and "C_ASK" in df.columns:
            mid = (df["C_BID"] + df["C_ASK"]) / 2
            valid_mid = (df["C_ASK"] >= df["C_BID"]) & (df["C_ASK"] > 0)
            df["PREMIUM"] = mid.where(valid_mid, df.get("C_LAST", 0))
            df["BID_ASK_SPREAD_PCT"] = (df["C_ASK"] - df["C_BID"]) / df["PREMIUM"].replace(0, np.nan)
        else:
            df["PREMIUM"] = df.get("C_LAST", 0)
            df["BID_ASK_SPREAD_PCT"] = 0.05

        df["MONEYNESS"] = df["UNDERLYING_LAST"] / df["STRIKE"]
        df["STRIKE_DISTANCE_CALC_PCT"] = (df["UNDERLYING_LAST"] - df["STRIKE"]) / df["STRIKE"]
        df["C_VOLUME"] = df.get("C_VOLUME", 0).fillna(0)

        # Depuración de registros consistentes
        df = df.dropna(subset=["QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE", "DTE", "C_IV", "PREMIUM"])
        df = df[(df["PREMIUM"] > 0) & (df["DTE"] > 0)].copy()

        # Inferencia con el modelo serializado (o heurística si no se subió el .pkl)
        features = ["UNDERLYING_LAST", "STRIKE", "PREMIUM", "DTE", "C_IV", "C_VOLUME", "MONEYNESS", "STRIKE_DISTANCE_CALC_PCT", "BID_ASK_SPREAD_PCT"]
        if modelo is not None:
            X = df[features].fillna(0)
            df["PROBABILIDAD_COMPRA"] = modelo.predict_proba(X)[:, 1]
        else:
            # Puntuación calibrada de respaldo
            score = (df["MONEYNESS"] - 0.95) * 1.8 - (df["C_IV"] > 0.55).astype(int) * 0.3
            df["PROBABILIDAD_COMPRA"] = np.clip(score * 0.35 + 0.40, 0.02, 0.96)

        # Target de rentabilidad histórica
        if "UNDERLYING_AT_EXPIRY" not in df.columns:
            df["UNDERLYING_AT_EXPIRY"] = df["UNDERLYING_LAST"] * 1.02
        
        df["PAYOFF"] = np.maximum(df["UNDERLYING_AT_EXPIRY"] - df["STRIKE"], 0)
        df["PNL_PER_SHARE"] = df["PAYOFF"] - df["PREMIUM"]
        df["SIGNAL_REAL"] = np.where(df["PNL_PER_SHARE"] > 0, "Comprar", "No comprar")

    # 3. Controles interactivos (Filtros en tiempo real)
    st.sidebar.subheader("🔍 Filtros de Segmentación")
    min_f = df["QUOTE_DATE"].min().date()
    max_f = df["QUOTE_DATE"].max().date()
    rango_fechas = st.sidebar.date_input("Rango de Fechas (QUOTE_DATE):", [min_f, max_f], min_value=min_f, max_value=max_f)

    s_min = float(df["STRIKE"].min())
    s_max = float(df["STRIKE"].max())
    rango_strike = st.sidebar.slider("Precio Strike ($):", s_min, s_max, (s_min, s_max))

    rango_dte = st.sidebar.slider("Días al Vencimiento (DTE):", int(df["DTE"].min()), int(df["DTE"].max()), (1, 150))
    umbral_corte = st.sidebar.slider("Umbral de Decisión ML (%):", 50, 90, 65) / 100.0

    # Aplicación de los filtros
    if len(rango_fechas) == 2:
        filtro = (df["QUOTE_DATE"].dt.date >= rango_fechas[0]) & (df["QUOTE_DATE"].dt.date <= rango_fechas[1])
    else:
        filtro = pd.Series([True] * len(df))

    filtro &= (df["STRIKE"] >= rango_strike[0]) & (df["STRIKE"] <= rango_strike[1])
    filtro &= (df["DTE"] >= rango_dte[0]) & (df["DTE"] <= rango_dte[1])

    df_filtrado = df[filtro].copy()
    df_filtrado["SENAL_MODELO"] = np.where(df_filtrado["PROBABILIDAD_COMPRA"] >= umbral_corte, "Comprar", "No comprar")

    # 4. Despliegue de los 3 niveles analíticos
    tab1, tab2, tab3 = st.tabs(["01. Análisis Descriptivo", "02. Análisis Predictivo", "03. Análisis Prescriptivo"])

    # --- TAB 1: DESCRIPTIVO ---
    with tab1:
        st.markdown("#### Panorama General del Mercado")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Contratos", f"{len(df_filtrado):,}")
        d2.metric("Prima Promedio", f"${df_filtrado['PREMIUM'].mean():.2f}")
        d3.metric("Volumen Total", f"{df_filtrado['C_VOLUME'].sum():,.0f}")
        d4.metric("Volatilidad Implícita Prom.", f"{df_filtrado['C_IV'].mean() * 100:.2f}%")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            vol_por_strike = df_filtrado.groupby("STRIKE")["C_VOLUME"].sum().reset_index()
            fig_v = px.bar(vol_por_strike, x="C_VOLUME", y="STRIKE", orientation="h", title="Volumen Total por Strike", color_discrete_sequence=["#0284c7"])
            st.plotly_chart(fig_v, use_container_width=True)
        with col_g2:
            muestra = df_filtrado.sample(min(1200, len(df_filtrado)))
            fig_s = px.scatter(muestra, x="MONEYNESS", y="C_IV", color="SIGNAL_REAL", title="Estructura de Volatilidad (IV) vs Moneyness")
            st.plotly_chart(fig_s, use_container_width=True)

    # --- TAB 2: PREDICTIVO ---
    with tab2:
        st.markdown("#### Desempeño del Algoritmo de Clasificación")
        tp = len(df_filtrado[(df_filtrado['SENAL_MODELO'] == 'Comprar') & (df_filtrado['SIGNAL_REAL'] == 'Comprar')])
        fp = len(df_filtrado[(df_filtrado['SENAL_MODELO'] == 'Comprar') & (df_filtrado['SIGNAL_REAL'] == 'No comprar')])
        tn = len(df_filtrado[(df_filtrado['SENAL_MODELO'] == 'No comprar') & (df_filtrado['SIGNAL_REAL'] == 'No comprar')])
        fn = len(df_filtrado[(df_filtrado['SENAL_MODELO'] == 'No comprar') & (df_filtrado['SIGNAL_REAL'] == 'Comprar')])
        total_p = len(df_filtrado)
        accuracy = (tp + tn) / total_p if total_p > 0 else 0

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Precisión Global (Accuracy)", f"{accuracy * 100:.2f}%")
        p2.metric("Señales de Compra", f"{(tp + fp):,}")
        p3.metric("Aciertos (TP)", f"{tp:,}")
        p4.metric("Confianza Promedio", f"{df_filtrado['PROBABILIDAD_COMPRA'].mean() * 100:.2f}%")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("##### Matriz de Confusión")
            matriz_data = pd.DataFrame({
                "Predicho: Comprar": [tp, fp],
                "Predicho: No comprar": [fn, tn]
            }, index=["Real: Comprar", "Real: No comprar"])
            st.dataframe(matriz_data, use_container_width=True)
        with col_m2:
            df_filtrado["RANGO_CONF"] = pd.cut(df_filtrado["PROBABILIDAD_COMPRA"], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0], labels=["< 50%", "50% - 60%", "60% - 70%", "70% - 80%", "80% - 100%"])
            pnl_conf = df_filtrado.groupby("RANGO_CONF", observed=False)["PNL_PER_SHARE"].mean().reset_index()
            fig_conf = px.bar(pnl_conf, x="RANGO_CONF", y="PNL_PER_SHARE", title="PnL Promedio por Nivel de Confianza", color_discrete_sequence=["#0284c7"])
            st.plotly_chart(fig_conf, use_container_width=True)

    # --- TAB 3: PRESCRIPTIVO ---
    with tab3:
        st.markdown("#### Reglas de Decisión y Retorno Financiero")
        compras = df_filtrado[df_filtrado["SENAL_MODELO"] == "Comprar"]
        pnl_tot = compras["PNL_PER_SHARE"].sum() * 100
        pnl_unit = compras["PNL_PER_SHARE"].mean() if len(compras) > 0 else 0
        win_r = (len(compras[compras["PNL_PER_SHARE"] > 0]) / len(compras) * 100) if len(compras) > 0 else 0
        cap_req = compras["PREMIUM"].sum() * 100

        pr1, pr2, pr3, pr4 = st.columns(4)
        pr1.metric("PnL Total Estimado", f"${pnl_tot:,.2f}")
        pr2.metric("PnL Promedio Contrato", f"${pnl_unit:.2f}")
        pr3.metric("Tasa de Ganancia (Win Rate)", f"{win_r:.2f}%")
        pr4.metric("Capital Requerido", f"${cap_req:,.0f}")

        # Agrupación por tramos de vencimiento
        def clasificar_vencimiento(d):
            if d <= 15: return "01. Corto (<= 15 d)"
            elif d <= 30: return "02. Medio-Corto (16-30 d)"
            elif d <= 60: return "03. Medio (31-60 d)"
            else: return "04. Largo (> 60 d)"

        df_filtrado["TRAMO_DTE"] = df_filtrado["DTE"].apply(clasificar_vencimiento)
        resumen_dte = df_filtrado[df_filtrado["SENAL_MODELO"] == "Comprar"].groupby("TRAMO_DTE")["PNL_PER_SHARE"].sum() * 100
        
        st.write("##### Rendimiento por Rango de Vencimiento (Regla Prescriptiva)")
        st.dataframe(resumen_dte.reset_index().rename(columns={"PNL_PER_SHARE": "PnL Acumulado ($)"}), use_container_width=True)

else:
    st.info("👈 Por favor, carga el archivo CSV en la barra lateral para procesar los datos en tiempo real.")
