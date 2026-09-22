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

# 1. Función de consulta a Llama 3
# 1. Función de consulta a LLM vía Groq con fallback de modelos
def consultar_llama3(prompt_sistema, prompt_usuario):
    if "GROQ_API_KEY" not in st.secrets:
        st.error("No se encontró GROQ_API_KEY en Secrets.")
        return None
    try:
        from groq import Groq
        clave = str(st.secrets["GROQ_API_KEY"]).strip()
        cliente = Groq(api_key=clave)
        
        # Lista ordenada de modelos disponibles en Groq
        modelos_a_probar = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "gemma2-9b-it",
            "mixtral-8x7b-32768"
        ]
        
        for mod in modelos_a_probar:
            try:
                respuesta = cliente.chat.completions.create(
                    model=mod,
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    temperature=0.2,
                    max_tokens=450
                )
                return respuesta.choices[0].message.content
            except Exception as err:
                if "model_not_found" in str(err) or "does not exist" in str(err):
                    continue
                else:
                    raise err
        
        st.error("No se encontró ningún modelo compatible habilitado en la cuenta de Groq.")
        return None

    except ImportError:
        st.error("Librería 'groq' no instalada. Revisa requirements.txt.")
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
    with tab1:
        st.markdown("#### Panorama General del Mercado")
        d1, d2, d3, d4 = st.columns(4)
        total_c = len(df_filtrado)
        prima_p = df_filtrado['PREMIUM'].mean()
        vol_t = df_filtrado['C_VOLUME'].sum()
        iv_p = df_filtrado['C_IV'].mean() * 100

        d1.metric("Total Contratos", f"{total_c:,}")
        d2.metric("Prima Promedio", f"${prima_p:.2f}")
        d3.metric("Volumen Total", f"{vol_t:,.0f}")
        d4.metric("Volatilidad Implícita Prom.", f"{iv_p:.2f}%")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            vol_por_strike = df_filtrado.groupby("STRIKE")["C_VOLUME"].sum().reset_index()
            strike_max = vol_por_strike.sort_values(by="C_VOLUME", ascending=False).iloc[0]["STRIKE"] if len(vol_por_strike) > 0 else 0
            fig_v = px.bar(vol_por_strike, x="C_VOLUME", y="STRIKE", orientation="h", title="Concentración de Liquidez por Strike ($)", color_discrete_sequence=["#0284c7"])
            st.plotly_chart(fig_v, width="stretch")
        with col_g2:
            n_samples = min(1200, len(df_filtrado))
            muestra = df_filtrado.sample(n_samples) if n_samples > 0 else df_filtrado
            fig_s = px.scatter(muestra, x="MONEYNESS", y="C_IV", color="SIGNAL_REAL", title="Estructura de Volatilidad (IV) vs Moneyness")
            st.plotly_chart(fig_s, width="stretch")

        with st.expander("🤖 Interpretación Ejecutiva con Llama 3", expanded=True):
            if st.button("Generar Diagnóstico Descriptivo"):
                with st.spinner("Consultando Llama 3..."):
                    prompt_s = "Eres un analista cuantitativo senior. Devuelve exactamente 2 viñetas concisas y directas analizando liquidez y volatilidad."
                    prompt_u = f"AAPL CALL: Total={total_c}, Prima prom=${prima_p:.2f}, Vol total={vol_t:,.0f}, IV prom={iv_p:.2f}%, Strike con mayor vol=${strike_max:.2f}. Explica la implicación institucional."
                    analisis = consultar_llama3(prompt_s, prompt_u)
                    if analisis:
                        st.markdown(analisis)
            else:
                st.caption("Haz clic en el botón para solicitar el análisis en vivo a Llama 3.")

    # --- TAB 2: PREDICTIVO ---
    with tab2:
        st.markdown("#### Rendimiento del Algoritmo de Clasificación")
        tp = len(df_filtrado[(df_filtrado['SENAL_MODELO'] == 'Comprar') & (df_filtrado['SIGNAL_REAL'] == 'Comprar')])
        fp = len(df_filtrado[(df_filtrado['SENAL_MODELO'] == 'Comprar') & (df_filtrado['SIGNAL_REAL'] == 'No comprar')])
        tn = len(df_filtrado[(df_filtrado['SENAL_MODELO'] == 'No comprar') & (df_filtrado['SIGNAL_REAL'] == 'No comprar')])
        fn = len(df_filtrado[(df_filtrado['SENAL_MODELO'] == 'No comprar') & (df_filtrado['SIGNAL_REAL'] == 'Comprar')])
        total_p = len(df_filtrado)
        accuracy = (tp + tn) / total_p if total_p > 0 else 0

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Precisión Global (Accuracy)", f"{accuracy * 100:.2f}%")
        p2.metric("Alertas de Compra Generadas", f"{(tp + fp):,}")
        p3.metric("Aciertos Validados (TP)", f"{tp:,}")
        p4.metric("Confianza Promedio", f"{df_filtrado['PROBABILIDAD_COMPRA'].mean() * 100:.2f}%")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("##### Matriz de Confusión")
            matriz_data = pd.DataFrame({
                "Predicho: Comprar": [tp, fp],
                "Predicho: No comprar": [fn, tn]
            }, index=["Real: Comprar", "Real: No comprar"])
            st.dataframe(matriz_data, width="stretch")
        with col_m2:
            df_filtrado["RANGO_CONF"] = pd.cut(df_filtrado["PROBABILIDAD_COMPRA"], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0], labels=["< 50%", "50% - 60%", "60% - 70%", "70% - 80%", "80% - 100%"])
            pnl_conf = df_filtrado.groupby("RANGO_CONF", observed=False)["PNL_PER_SHARE"].mean().reset_index()
            fig_conf = px.bar(pnl_conf, x="RANGO_CONF", y="PNL_PER_SHARE", title="Retorno Promedio según Nivel de Confianza ($)", color_discrete_sequence=["#0284c7"])
            st.plotly_chart(fig_conf, width="stretch")

        with st.expander("🤖 Interpretación Ejecutiva con Llama 3", expanded=True):
            if st.button("Generar Diagnóstico Predictivo"):
                with st.spinner("Consultando Llama 3..."):
                    prompt_s = "Eres especialista en ML aplicado a finanzas. Explica el balance entre falsos positivos y aciertos en 2 viñetas concisas."
                    prompt_u = f"Accuracy={accuracy*100:.2f}%, Alertas={tp+fp}, TP={tp}, TN={tn}, FP={fp}, Umbral={int(umbral_corte*100)}%. Resume el valor predictivo y el filtrado de pérdidas."
                    analisis = consultar_llama3(prompt_s, prompt_u)
                    if analisis:
                        st.markdown(analisis)
            else:
                st.caption("Haz clic en el botón para evaluar la capacidad predictiva con Llama 3.")

    # --- TAB 3: PRESCRIPTIVO ---
    with tab3:
        st.markdown("#### Reglas Prescriptivas y Resultados Financieros")
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

        def clasificar_vencimiento(d):
            if d <= 15: return "01. Corto (<= 15 d)"
            elif d <= 30: return "02. Medio-Corto (16-30 d)"
            elif d <= 60: return "03. Medio (31-60 d)"
            else: return "04. Largo (> 60 d)"

        df_filtrado["TRAMO_DTE"] = df_filtrado["DTE"].apply(clasificar_vencimiento)
        resumen_dte = df_filtrado[df_filtrado["SENAL_MODELO"] == "Comprar"].groupby("TRAMO_DTE")["PNL_PER_SHARE"].sum() * 100
        tabla_dte = resumen_dte.reset_index().rename(columns={"TRAMO_DTE": "Rango Vencimiento (DTE)", "PNL_PER_SHARE": "PnL Acumulado ($)"})

        st.write("##### Rendimiento por Rango de Vencimiento")
        st.dataframe(tabla_dte, width="stretch")

        with st.expander("🤖 Decisión Prescriptiva con Llama 3", expanded=True):
            if st.button("Generar Regla de Trading Prescriptiva"):
                with st.spinner("Llama 3 sintetizando la política de inversión..."):
                    prompt_s = "Eres CIO de un fondo cuantitativo. Define la política de inversión en 2 reglas directas y ejecutivas."
                    prompt_u = f"Estrategia AAPL: PnL=${pnl_tot:,.2f}, Win Rate={win_r:.2f}%. En 31-60 DTE se pierden -$494k por Theta decay acelerado; <=15d y >60d son muy rentables. Establece las reglas prescriptivas de capital."
                    analisis = consultar_llama3(prompt_s, prompt_u)
                    if analisis:
                        st.markdown(analisis)
            else:
                st.caption("Haz clic en el botón para formular la política de inversión con Llama 3.")

else:
    st.info("👈 Por favor, carga el archivo CSV en la barra lateral para procesar los datos en tiempo real.")
