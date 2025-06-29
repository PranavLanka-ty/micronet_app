# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:43:10 2025
Cleaned version: reads interaction matrices and generates combined interaction graph.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

def load_excel(file):
    """Load matrices and plotting settings from Excel."""
    pos_df = pd.read_excel(file, sheet_name='Positive', header=None)
    neg_df = pd.read_excel(file, sheet_name='Negative', header=None)
    settings_df = pd.read_excel(file, sheet_name='PlotSettings', header=None)
    return pos_df, neg_df, settings_df

def extract_settings(settings_df):
    """Extract plotting parameters from settings sheet into a dictionary."""
    def get(key, default=None, cast=str):
        val = settings_df.loc[settings_df[0] == key, 1]
        return cast(val.values[0]) if not val.empty else default

    return {
        "scale neutral strength": get('scale neutral strength', 1.0, float),
        "scale linewidth": get('scale linewidth', 1.0, float),
        "scale arrowsize": get('scale arrowsize', 1.0, float),
        "combined transparency": get('combined transparency', 0.5, float),
        "nodeSize": get('nodeSize', 8, int),
        "nodeFontSize": get('nodeFontSize', 12, int),
        "nodeColor": get('nodeColor', 'skyblue', str),
        "Positive Line Color": get('Positive Line Color', 'green', str),
        "Positive Arrow Color": get('Positive Arrow Color', 'darkgreen', str),
        "Negative Line Color": get('Negative Line Color', 'red', str),
        "Negative Arrow Color": get('Negative Arrow Color', 'darkred', str),
        "Neutral Color": get('Neutral Color', 'gray', str),
    }

def extract_matrix(raw_df):
    """Extract node names and adjacency matrix from sheet."""
    node_names = raw_df.iloc[0, 1:].tolist()
    N = len(node_names)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            val = raw_df.iloc[i + 1, j + 1]
            W[i, j] = float(val) if not pd.isna(val) else 0.0
    return node_names, W

def plot_graph(W, W_other, node_names, filename, pos, params, inttype):
    """Draw directed graph with arrows scaled by weight, then save."""
    baseline = {
        'factor': 1e-4,
        'width': 0.1,
        'arrow': 4
    }

    # Color selection based on interaction type
    if inttype == 1:
        edge_color = params["Positive Line Color"]
        arrow_color = params["Positive Arrow Color"]
    else:
        edge_color = params["Negative Line Color"]
        arrow_color = params["Negative Arrow Color"]
    neutral_color = params["Neutral Color"]

    # Add baseline to help visualize absent edges
    W_display = W + baseline['factor'] * (np.ones_like(W) - np.eye(len(W)))

    # Construct directed graph
    G = nx.DiGraph()
    for i, src in enumerate(node_names):
        for j, dst in enumerate(node_names):
            if W_display[i, j] != 0:
                G.add_edge(dst, src, weight=W_display[i, j])

    # Weight scaling
    weights = np.array([abs(data['weight']) for _, _, data in G.edges(data=True)])
    max_weight = max(weights.max(), 1)
    linewidths = baseline['width'] + params["scale linewidth"] * (weights / max_weight)
    arrowsizes = baseline['arrow'] + params["scale arrowsize"] * (weights / max_weight)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=params["nodeSize"],
                           node_color=params["nodeColor"])

    # Draw node labels
    for node, (x, y) in pos.items():
        ax.text(x, y + 0.06, node, fontsize=params["nodeFontSize"],
                ha='center', va='center', fontstyle='italic',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    # Draw curved arrows with strength-scaled width and size
    for (u, v), lw, arrow_size in zip(G.edges(), linewidths, arrowsizes):
        if u == v: continue
        i, j = node_names.index(v), node_names.index(u)

        is_neutral = np.isclose(W_display[i, j], baseline['factor'], rtol=1e-5)
        other_strength = W_other[i, j]

        color = (
            (1, 1, 1, 0) if is_neutral and not np.isclose(other_strength, 0)
            else (neutral_color if is_neutral else edge_color)
        )

        patch = FancyArrowPatch(pos[u], pos[v],
                                connectionstyle="arc3,rad=0.2",
                                arrowstyle='-',
                                color=color,
                                linewidth=lw * (params["scale neutral strength"] if is_neutral else 1),
                                mutation_scale=arrow_size)
        ax.add_patch(patch)

        # Add arrowhead only for non-neutral edges
        if not is_neutral:
            ctrl_x = (pos[u][0] + pos[v][0]) / 2 + 0.2 * (pos[v][1] - pos[u][1])
            ctrl_y = (pos[u][1] + pos[v][1]) / 2 - 0.2 * (pos[v][0] - pos[u][0])
            t = 0.5
            mid_x = (1 - t)**2 * pos[u][0] + 2 * (1 - t) * t * ctrl_x + t**2 * pos[v][0]
            mid_y = (1 - t)**2 * pos[u][1] + 2 * (1 - t) * t * ctrl_y + t**2 * pos[v][1]
            dx, dy = 0.01 , 0.01 

            mid_arrow = FancyArrowPatch(pos[u], (mid_x + dx, mid_y + dy),
            # mid_arrow = FancyArrowPatch( (mid_x - dx * 0.01, mid_y - dy * 0.01),
            #                              (mid_x + dx * 0.01, mid_y + dy * 0.01),
                                         connectionstyle="arc3,rad=0.1",
                                         arrowstyle='-|>',
                                         color=arrow_color,
                                         linewidth=0,
                                         mutation_scale=arrow_size * 3)
            ax.add_patch(mid_arrow)
        

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, pad_inches=0)
    plt.close()

def generate_all_graphs(excel_path):
    """Load data, plot both positive/negative graphs, and combine images."""
    df_pos, df_neg, df_settings = load_excel(excel_path)
    params = extract_settings(df_settings)

    node_names, W_pos = extract_matrix(df_pos)
    _, W_neg = extract_matrix(df_neg)

    assert node_names == _, "Mismatch in species names between sheets."

    # Use unified layout
    G_all = nx.DiGraph()
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if W_pos[i, j] + W_neg[i, j] != 0:
                G_all.add_edge(node_names[j], node_names[i])
    pos = nx.circular_layout(G_all)

    plot_graph(W_pos, W_neg, node_names, "fig1.png", pos, params, inttype=1)
    plot_graph(W_neg, W_pos, node_names, "fig2.png", pos, params, inttype=-1)

    # Blend the two figures
    img1 = Image.open("fig1.png").convert("RGBA")
    img2 = Image.open("fig2.png").convert("RGBA")
    img2 = Image.blend(Image.new("RGBA", img2.size, (255, 255, 255, 0)), img2, params["combined transparency"])

    final_img = Image.alpha_composite(img1, img2)
    final_img.convert("RGB").save("combinedFigures.jpg")

    return "fig1.png", "fig2.png", "combinedFigures.jpg"

# Execute the graph generation from Excel
if __name__ == "__main__":
    fig1, fig2, combined = generate_all_graphs("S3- MicroNet Template.xlsx")
