import streamlit as st
from micronetplot_core import generate_all_graphs
from PIL import Image
import os

# Set app layout
st.set_page_config(page_title="Sihumi Digraph Plotter", layout="wide")

st.title("MICRONETPLOT")
st.markdown("Upload the prepared Excel file to visualize the Microbial interaction network.")

# Upload file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # Save the uploaded file
    config_path = "interaction_config.xlsx"
    with open(config_path, "wb") as f:
        f.write(uploaded_file.read())

    # Generate graphs
    with st.spinner("Generating plots..."):
        fig1_path, fig2_path, combined_path = generate_all_graphs(config_path)

    st.success("Done!")

    # Load the generated images as PIL Images
    img1 = Image.open(fig1_path)
    img2 = Image.open(fig2_path)
    img_combined = Image.open(combined_path)

    # Display individual graphs side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Positive Interactions", width=600)
    with col2:
        st.image(img2, caption="Negative Interactions", width=600)

    # Show combined figure with full width
    st.image(img_combined, caption="Overlaid Result", width=1200)

    # Optional download button for the high-resolution image
    with open(combined_path, "rb") as f:
        st.download_button("📥 Download Combined PNG", f, file_name="combinedInteractions.png")
