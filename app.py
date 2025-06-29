# app.py
import streamlit as st
from micronetplot_core import generate_all_graphs

st.set_page_config(page_title="Sihumi Digraph Plotter", layout="wide")
st.title("MICRONETPLOT")
st.markdown("Upload the prepared Excel file to visualize the Microbial network.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # Save uploaded file temporarily
    with open("interaction_config.xlsx", "wb") as f:
        f.write(uploaded_file.read())

    # Generate figures
    with st.spinner("Generating plots..."):
        fig1, fig2, fig2_dummy, combined = generate_all_graphs("interaction_config.xlsx")

    st.success("Done!")

    # Display results in columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(fig1, caption="Positive Interactions", use_container_width=True)
    with col2:
        st.image(fig2, caption="Negative Interactions", use_container_width=True)

    # Combined figure
    st.image(combined, caption="Overlaid Result (Pos + Neg)", use_container_width=True)
    st.image(fig2_dummy, caption="Dummy Negative Interaction Graph")

    # # # Optional debug display
    # with st.expander("Show dummy fig2 (used for blending)"):
    #     st.image(fig2_dummy, caption="Dummy Negative Interaction Graph")
