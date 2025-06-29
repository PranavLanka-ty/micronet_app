# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:43:55 2025

@author: Pranav Lanka
"""

import streamlit as st
from micronetplot_core import generate_all_graphs
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Sihumi Digraph Plotter", layout="wide")

st.title("MICRONETPLOT")
st.markdown("Upload the prepared Excel file to visualize the Microbial network.")

def convert_image_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')  # Save figure to buffer
    buf.seek(0)
    return Image.open(buf)  # Open it as PIL image


uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    with open("interaction_config.xlsx", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Generating plots..."):
        fig1, fig2, fig2_dummy, combined = generate_all_graphs("interaction_config.xlsx")

    st.success("Done!")

    fig1 = fig_to_pil(fig1)
    fig2 = fig_to_pil(fig2)
    combined = fig_to_pil(combined)
    # Prepare image byte buffers
    fig1_bytes = convert_image_to_bytes(fig1)
    fig2_bytes = convert_image_to_bytes(fig2)
    combined_bytes = convert_image_to_bytes(combined)

    col1, col2 = st.columns(2)
    with col1:
        st.image(fig1, caption="Positive Interactions", use_container_width=True)
        st.download_button("Download Positive", fig1_bytes, file_name="positive_interactions.png", mime="image/png")

    with col2:
        st.image(fig2, caption="Negative Interactions", use_container_width=True)
        st.download_button("Download Negative", fig2_bytes, file_name="negative_interactions.png", mime="image/png")

    st.image(combined, caption="Overlaid Result", use_container_width=True)
    st.download_button("Download Overlaid", combined_bytes, file_name="combined_interactions.png", mime="image/png")
