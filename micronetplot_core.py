# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 13:53:53 2025

@author: Pranav Lanka
"""

"""
Updated: Generates positive and negative interaction graphs with transparency masks for precise blending.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from PIL import Image

def load_excel(file):
    pos_df = pd.read_excel(file, sheet_name='Positive', header=None)
    neg_df = pd.read_excel(file, sheet_name='Negative', header=None)
    settings_df = pd.read_excel(file, sheet_name='PlotSettings', header=None)
    return pos_df, neg_df, settings_df

def extract_settings(settings_df):
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
        "Positive Line Color": get('Positive Line Color', 'orange', str),
        "Positive Arrow Color": get('Positive Arrow Color', 'chocolate', str),
        "Negative Line Color": get('Negative Line Color', 'teal', str),
        "Negative Arrow Color": get('Negative Arrow Color', 'darkslategray', str),
        "Neutral Color": get('Neutral Color', 'gray', str),
    }

def extract_matrix(raw_df):
    node_names = raw_df.iloc[0, 1:].tolist()
    N = len(node_names)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            val = raw_df.iloc[i + 1, j + 1]
            W[i, j] = float(val) if not pd.isna(val) else 0.0
    return node_names, W

def plot_graph(W_pos, W_neg, node_names, filename, pos, params, highlight='positive'):
    baseline = {'factor': 1e-4, 'width': 0.1, 'arrow': 4}
    N = len(node_names)
    W_combined = W_pos + W_neg
    W_display = W_combined + baseline['factor'] * (np.ones_like(W_combined) - np.eye(N))

    G = nx.DiGraph()
    for i, src in enumerate(node_names):
        for j, dst in enumerate(node_names):
            if W_display[i, j] != 0:
                G.add_edge(dst, src, weight=W_display[i, j])

    weights = np.array([abs(data['weight']) for _, _, data in G.edges(data=True)])
    max_weight = max(weights.max(), 1)
    linewidths = baseline['width'] + params["scale linewidth"] * (weights / max_weight)
    arrowsizes = baseline['arrow'] + params["scale arrowsize"] * (weights / max_weight)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=params["nodeSize"],
                           node_color=params["nodeColor"])
    for node, (x, y) in pos.items():
        ax.text(x, y + 0.06, node, fontsize=params["nodeFontSize"],
                ha='center', va='center', fontstyle='italic',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    for (u, v), lw, arrow_size in zip(G.edges(), linewidths, arrowsizes):
        if u == v:
            continue
        i, j = node_names.index(v), node_names.index(u)

        is_positive = W_pos[i, j] != 0
        is_negative = W_neg[i, j] != 0
        is_neutral = not is_positive and not is_negative
        
        # Default: visible neutral color
        edge_color = params["Neutral Color"]
        arrow_color = (1, 1, 1, 0) 
        
        # If not neutral, assign correct type
        if is_positive:
            edge_color = params["Positive Line Color"]
            arrow_color = params["Positive Arrow Color"]
        if is_negative:
            edge_color = params["Negative Line Color"]
            arrow_color = params["Negative Arrow Color"]
        
        # Transparency logic: mask the non-highlighted interaction
        if (highlight == 'positive' and is_negative) or (highlight == 'negative' and is_positive):
            edge_color = (1, 1, 1, 0)
            arrow_color = (1, 1, 1, 0)


        x_start, y_start = pos[u]
        x_end, y_end = pos[v]
        ctrl_x = (x_start + x_end) / 2 + 0.2 * (y_end - y_start)
        ctrl_y = (y_start + y_end) / 2 - 0.2 * (x_end - x_start)

        patch = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                connectionstyle="arc3,rad=0.2",
                                arrowstyle='-',
                                color=edge_color,
                                linewidth=lw,
                                mutation_scale=arrow_size)
        ax.add_patch(patch)

        # Add centered arrowhead
        t = 0.5
        mid_x = (1 - t) ** 2 * x_start + 2 * (1 - t) * t * ctrl_x + t ** 2 * x_end
        mid_y = (1 - t) ** 2 * y_start + 2 * (1 - t) * t * ctrl_y + t ** 2 * y_end
        dx = 2 * (1 - t) * (ctrl_x - x_start) + 2 * t * (x_end - ctrl_x)
        dy = 2 * (1 - t) * (ctrl_y - y_start) + 2 * t * (y_end - ctrl_y)
        norm = np.hypot(dx, dy)
        dx, dy = (dx / norm, dy / norm) if norm != 0 else (1.0, 0.0)

        arrow = FancyArrowPatch(
            (mid_x - dx * 0.01, mid_y - dy * 0.01),
            (mid_x + dx * 0.01, mid_y + dy * 0.01),
            connectionstyle="arc3,rad=0",
            arrowstyle='-|>',
            color=arrow_color,
            linewidth=0,
            mutation_scale=arrow_size * 3,
            zorder=10
        )
        ax.add_patch(arrow)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.1) #, transparent=True
    plt.close()

def generate_all_graphs(excel_path):
    df_pos, df_neg, df_settings = load_excel(excel_path)
    params = extract_settings(df_settings)
    node_names, W_pos = extract_matrix(df_pos)
    _, W_neg = extract_matrix(df_neg)

    assert node_names == _, "Node names mismatch between sheets."

    G_all = nx.DiGraph()
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if W_pos[i, j] + W_neg[i, j] != 0:
                G_all.add_edge(node_names[j], node_names[i])
    pos = nx.circular_layout(G_all)

    plot_graph(W_pos, W_neg, node_names, "fig1.png", pos, params, highlight='positive')
    plot_graph(W_pos, W_neg, node_names, "fig2.png", pos, params, highlight='negative')
    
    img1 = Image.open("fig1.png").convert("RGBA")
    img2 = Image.open("fig2.png").convert("RGBA")

    # Force exact same size before blending
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, resample=Image.BICUBIC)

    alpha = params["combined transparency"]
    img2_masked = Image.blend(Image.new("RGBA", img2.size, (255, 255, 255, 0)), img2, alpha)
    
    combined = Image.alpha_composite(img1, img2_masked)
    combined.convert("RGB").save("combinedFigures.jpg", quality=95)


    return "fig1.png", "fig2.png", "combinedFigures.jpg"

#%%
# # Run the graph generator
fig1, fig2, combined = generate_all_graphs("S3- MicroNet Template.xlsx")
