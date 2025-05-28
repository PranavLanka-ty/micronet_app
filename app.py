# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:43:55 2025

@author: BioPixS
"""

import streamlit as st
from micronetplot_core import generate_all_graphs

st.set_page_config(page_title="Sihumi Digraph Plotter", layout="wide")

st.title("MICRONETPLOT")
st.markdown("Upload the prepared Excel file to visualize the Microbial network.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    with open("interaction_config.xlsx", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Generating plots..."):
        fig1, fig2, combined = generate_all_graphs("interaction_config.xlsx")

    st.success("Done!")

    col1, col2 = st.columns(2)
    with col1:
        st.image(fig1, caption="Positive Interactions", use_container_width =True)
    with col2:
        st.image(fig2, caption="Negative Interactions", use_container_width =True)

    st.image(combined, caption="Overlayed Result", use_container_width =True)
