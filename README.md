# MicronetPlot: A Python Tool for Visualizing Microbial Interaction Networks

**Repository**: https://github.com/pranavlanka-ty/micronet_app  
**License**: CC  
**Author responsible for this code**: Pranav Lanka 
**Last Updated**: [2025-12-20]  

---

## Overview

MicronetPlot is a Python-based software designed for visualizing and analyzing microbial interaction matrices as directed edge-weighted graphs. It supports intuitive visual overlays of positive and negative interaction networks with customizable aesthetics and edge importance.

This package is intended to support scientific research involving microbial community modeling, including defined consortia like the **SIHUMI** (Simplified Human Intestinal Microbiota) model. It enables the generation of high-quality figures that can be embedded in scientific publications.

---

## Repository Contents

- `/app.py` – Main script for launching the web-based GUI.
- `/micronetplot_core.py` – Core plotting and network handling logic.
- `/assets/` – Includes images and style definitions.
- `/data/` – Sample datasets and interaction matrices.
- `/outputs/` – Output folder for generated plots.
- `/requirements.txt` – Python dependencies.

---

## System Requirements

- Python 3.8+
- Platform independent (tested on macOS, Linux, Windows)
- 4+ GB RAM recommended for large matrices

---

## Installation

```bash
git clone https://github.com/pranavlanka-ty/micronet_app.git
cd micronet_app
pip install -r requirements.txt
python app.py

---

```markdown
## Performance

**Typical install time**: ~1–2 minutes using `pip install -r requirements.txt` (may vary slightly by network speed and Python environment setup).
**Run time**: Once installed, generating a typical interaction graph from a properly formatted `.xlsx` file takes **under 5 seconds**.  
**Web app startup**: The Streamlit app launches locally within **1–2 seconds** after executing `python app.py`
