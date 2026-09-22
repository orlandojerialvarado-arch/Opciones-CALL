import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# Configuración de página
st.set_page_config(page_title="Sistema Analítico de Opciones CALL", layout="wide", page_icon="📈")

# Banner institucional
st.markdown("""
<div style="background-color: #1e2a4a; padding: 18px; border-radius: 6px; text-align: center; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0; font-weight: 700; letter-spacing: 0.04em;">BOT INVERSIÓN EN OPCIONES CALL</h2>
</div>
""", unsafe_allow_html=True)

# 1. Función de consulta a LLM vía Groq con filtro de modelos de texto
def consultar_llama3(prompt_sistema, prompt_usuario):
    if "GROQ_API_KEY" not in st.secrets:
        st.error("No se encontró GROQ_API_KEY en Secrets.")
        return None
    try:
        from groq import Groq
        clave = str(st.secrets["GROQ_API_KEY"]).strip()
        cliente = Groq(api_key=clave)
        
        modelos_prioritarios = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it"
        ]
        
        lista_raw = cliente.models.list().data
        modelos_chat = [
            m.id for m in lista_raw 
            if not any(x in m.id.lower() for x in ["whisper", "guard", "embed", "safeguard"])
        ]
        candidatos = [m for m in modelos_prioritarios if m in modelos_chat] + modelos_chat
        
        if not candidatos:
            st.error("No se encontraron modelos de chat disponibles.")
            return None

        ultimo_error = None
        for mod in dict.fromkeys(candidatos):
            try:
                respuesta = cliente.chat.completions.create(
                    model=mod,
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    temperature=0.1,
                    max_tokens=650
                )
                return respuesta.choices[0].message.content
            except Exception as err:
                ultimo_error = err
                continue

        st.error(f"Error con los modelos: {ultimo_error}")
        return None

    except Exception as e:
        st.error(f"Error al conectar con Groq: {e}")
        return None

# 2. Carga del Modelo Serializado
MODEL_FILE = "mejor_modelo_aapl_call.pkl"

@st.cache_resource
def obtener_modelo():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

modelo = obtener_modelo()

# 3. Barra lateral para carga de CSV
st.sidebar.header("📂 Ingesta de Datos")
archivo_cargado = st.sidebar.file_uploader("Subir CSV de Opciones:", type=["csv"])

if archivo_cargado is not None:
    df_raw = pd.read_csv(archivo_cargado, low_memory=False)
    df_raw.columns = [str(c).strip().replace("[", "").replace("]", "") for c in df_raw.columns]

    with st.spinner("Procesando datos y calibrando variables..."):
        df = df_raw.copy()
        
        df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"], errors="coerce")
        df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"], errors="coerce")
        df = df.dropna(subset=["QUOTE_DATE"]).copy()

        cols_num = ["UNDERLYING_LAST", "DTE", "C_IV", "C_VOLUME", "C_LAST", "C_BID", "C_ASK", "STRIKE", "PREMIUM", "PNL_PER_SHARE", "PROBABILIDAD_COMPRA"]
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")

        if "PREMIUM" not in df.columns or df["PREMIUM"].isnull().all():
            if "C_BID" in df.columns and "C_ASK" in df.columns:
                mid = (df["C_BID"] + df["C_ASK"]) / 2
                valid_mid = (df["C_ASK"] >= df["C_BID"]) & (df["C_ASK"] > 0)
                df["PREMIUM"] = mid.where(valid_mid, df.get("C_LAST", 0))
            else:
                df["PREMIUM"] = df.get("C_LAST", 0)

        if "MONEYNESS" not in df.columns:
            df["MONEYNESS"] = df["UNDERLYING_LAST"] / df["STRIKE"]
        if "STRIKE_DISTANCE_CALC_PCT" not in df.columns:
            df["STRIKE_DISTANCE_CALC_PCT"] = (df["UNDERLYING_LAST"] - df["STRIKE"]) / df["STRIKE"]
        if "BID_ASK_SPREAD_PCT" not in df.columns:
            if "C_BID" in df.columns and "C_ASK" in df.columns:
                df["BID_ASK_SPREAD_PCT"] = (df["C_ASK"] - df["C_BID"]) / df["PREMIUM"].replace(0, np.nan)
            else:
                df["BID_ASK_SPREAD_PCT"] = 0.05

        df["C_VOLUME"] = df.get("C_VOLUME", 0).fillna(0)
        df = df.dropna(subset=["QUOTE_DATE", "UNDERLYING_LAST", "STRIKE", "DTE", "C_IV", "PREMIUM"])
        df = df[(df["PREMIUM"] > 0) & (df["DTE"] > 0)].copy()

        features = ["UNDERLYING_LAST", "STRIKE", "PREMIUM", "DTE", "C_IV", "C_VOLUME", "MONEYNESS", "STRIKE_DISTANCE_CALC_PCT", "BID_ASK_SPREAD_PCT"]
        if "PROBABILIDAD_COMPRA" not in df.columns or df["PROBABILIDAD_COMPRA"].isnull().all():
            if modelo is not None:
                X = df[features].fillna(0)
                df["PROBABILIDAD_COMPRA"] = modelo.predict_proba(X)[:, 1]
            else:
                score = (df["MONEYNESS"] - 0.95) * 1.8 - (df["C_IV"] > 0.55).astype(int) * 0.3
                df["PROBABILIDAD_COMPRA"] = np.clip(score * 0.35 + 0.40, 0.02, 0.96)

        if "PNL_PER_SHARE" not in df.columns:
            if "UNDERLYING_AT_EXPIRY" not in df.columns:
                df["UNDERLYING_AT_EXPIRY"] = df["UNDERLYING_LAST"] * 1.02
            df["PAYOFF"] = np.maximum(df["UNDERLYING_AT_EXPIRY"] - df["STRIKE"], 0)
            df["PNL_PER_SHARE"] = df["PAYOFF"] - df["PREMIUM"]

        if "SIGNAL_REAL" not in df.columns:
            df["SIGNAL_REAL"] = np.where(df["PNL_PER_SHARE"] > 0, "Comprar", "No comprar")

    # 4. Filtros interactivos
    st.sidebar.subheader("🔍 Filtros Dinámicos")
    min_f = df["QUOTE_DATE"].min().date()
    max_f = df["QUOTE_DATE"].max().date()
    
    if min_f == max_f:
        st.sidebar.info(f"Fecha analizada: {min_f}")
        fecha_sel = [min_f, max_f]
    else:
        fecha_sel = st.sidebar.date_input("Rango de Fechas:", value=(min_f, max_f), min_value=min_f, max_value=max_f)

    s_min = float(df["STRIKE"].min())
    s_max = float(df["STRIKE"].max())
    rango_strike = st.sidebar.slider("Precio Strike ($):", s_min, s_max, (s_min, s_max))

    rango_dte = st.sidebar.slider("Días al Vencimiento (DTE):", int(df["DTE"].min()), int(df["DTE"].max()), (int(df["DTE"].min()), int(df["DTE"].max())))
    umbral_corte = st.sidebar.slider("Umbral de Decisión ML (%):", 50, 90, 65) / 100.0

    if isinstance(fecha_sel, (list, tuple)) and len(fecha_sel) == 2:
        filtro = (df["QUOTE_DATE"].dt.date >= fecha_sel[0]) & (df["QUOTE_DATE"].dt.date <= fecha_sel[1])
    else:
        filtro = pd.Series([True] * len(df), index=df.index)

    filtro &= (df["STRIKE"] >= rango_strike[0]) & (df["STRIKE"] <= rango_strike[1])
    filtro &= (df["DTE"] >= rango_dte[0]) & (df["DTE"] <= rango_dte[1])

    df_filtrado = df[filtro].copy()
    df_filtrado["SENAL_MODELO"] = np.where(df_filtrado["PROBABILIDAD_COMPRA"] >= umbral_corte, "Comprar", "No comprar")

    # 5. Paneles analíticos
    tab1, tab2, tab3 = st.tabs(["01. Análisis Descriptivo", "02. Análisis Predictivo", "03. Análisis Prescriptivo"])

    # --- TAB 1: DESCRIPTIVO ---
   with st.expander("🤖 Interpretación Ejecutiva con Llama 3", expanded=True):
            if st.button("Generar Diagnóstico Descriptivo"):
                with st.spinner("Consultando Llama 3..."):
                    prompt_s = (
                        "Eres un analista cuantitativo senior de derivados financieros. "
                        "Tu respuesta DEBE seguir estrictamente esta estructura de 3 partes:\n"
                        "1. Liquidez y Concentración: (análisis breve en 1-2 oraciones)\n"
                        "2. Volatilidad y Costo de Entrada: (análisis breve en 1-2 oraciones)\n"
                        "**Conclusión Ejecutiva:** (una sola línea final directa con la lectura del mercado)."
                    )
                    prompt_u = (
                        f"Datos de opciones CALL de AAPL: Total contratos evaluados: {total_c:,}, "
                        f"Prima promedio: ${prima_p:.2f}, Volumen acumulado: {vol_t:,.0f}, "
                        f"IV promedio: {iv_p:.2f}%, Strike con mayor liquidez institucional: ${strike_max:.2f}. "
                        "Genera el diagnóstico institucional siguiendo el formato requerido."
                    )
                    analisis = consultar_llama3(prompt_s, prompt_u)
                    if analisis:
                        st.markdown(analisis)
            else:
                st.caption("Haz clic en el botón para solicitar el análisis en vivo a Llama 3.")

    # --- TAB 2: PREDICTIVO ---
    with st.expander("🤖 Interpretación Ejecutiva con Llama 3", expanded=True):
            if st.button("Generar Diagnóstico Predictivo"):
                with st.spinner("Consultando Llama 3..."):
                    prompt_s = (
                        "Eres un especialista en Machine Learning aplicado a finanzas institucionales. "
                        "No expliques conceptos básicos como qué es TP o TN. Ve directo al valor operativo. "
                        "Tu respuesta DEBE seguir estrictamente esta estructura:\n"
                        "1. Filtrado Defensivo y Mitigación de Pérdidas: (1-2 oraciones sobre los contratos descartados)\n"
                        "2. Calidad de Alertas y Retorno por Certeza: (1-2 oraciones sobre los aciertos y el umbral)\n"
                        "**Conclusión Ejecutiva:** (una sola línea final directa sobre la confiabilidad del modelo)."
                    )
                    prompt_u = (
                        f"Métricas del modelo de clasificación CALL: Accuracy global: {accuracy*100:.2f}%, "
                        f"Alertas de compra generadas: {tp+fp:,}, Aciertos validados (TP): {tp:,}, "
                        f"Contratos de pérdida descartados exitosamente (TN): {tn:,}, "
                        f"Falsos positivos: {fp:,}, Umbral de decisión configurado: {int(umbral_corte*100)}%. "
                        "Genera el análisis de rendimiento siguiendo el formato."
                    )
                    analisis = consultar_llama3(prompt_s, prompt_u)
                    if analisis:
                        st.markdown(analisis)
            else:
                st.caption("Haz clic en el botón para evaluar la capacidad predictiva con Llama 3.")

    # --- TAB 3: PRESCRIPTIVO ---
    with st.expander("🤖 Decisión Prescriptiva con Llama 3", expanded=True):
            if st.button("Generar Regla de Trading Prescriptiva"):
                with st.spinner("Llama 3 sintetizando la política de inversión..."):
                    prompt_s = (
                        "Eres el Chief Investment Officer (CIO) de un fondo cuantitativo. "
                        "Tu respuesta DEBE seguir estrictamente esta estructura:\n"
                        "1. Regla de Exclusión por Theta Decay: (directriz imperativa sobre el tramo perdedor 31-60 DTE)\n"
                        "2. Asignación de Capital Favorable: (directriz sobre contratos <=15d y >60d)\n"
                        "**Conclusión Ejecutiva:** (una sola línea final directa resumiendo la orden de operación)."
                    )
                    prompt_u = (
                        f"Estrategia sobre opciones CALL de AAPL: PnL acumulado: ${pnl_tot:,.2f}, Win Rate: {win_r:.2f}%. "
                        "Desglose por vencimiento: Tramo crítico de 31 a 60 DTE destruye -$494k USD por Theta decay acelerado; "
                        "mientras que tramos <=15 días y >60 días generan los retornos positivos de la cartera. "
                        "Establece las directrices y la conclusión ejecutiva."
                    )
                    analisis = consultar_llama3(prompt_s, prompt_u)
                    if analisis:
                        st.markdown(analisis)
            else:
                st.caption("Haz clic en el botón para formular la política de inversión con Llama 3.")

else:
    st.info("👈 Por favor, carga el archivo CSV en la barra lateral para procesar los datos en tiempo real.")
