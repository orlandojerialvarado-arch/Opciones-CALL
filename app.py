import streamlit as st

st.title("Prueba de Servidor Python Activo 🚀")
st.write("¡Hola mundo! El motor de Python en la nube está corriendo.")

archivo = st.file_uploader("Sube una prueba de archivo CSV aquí", type=["csv"])
if archivo is not None:
    st.success(f"Archivo recibido exitosamente: {archivo.name}")
