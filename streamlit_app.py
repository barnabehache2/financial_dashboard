import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hello World Streamlit", page_icon="👋", layout="centered")

st.title("Hello world")
st.write("Bienvenue dans ma première app Streamlit.")

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="sin(x)"))
fig.update_layout(
    title="Courbe sinus",
    xaxis_title="x",
    yaxis_title="sin(x)",
)

st.plotly_chart(fig, use_container_width=True)