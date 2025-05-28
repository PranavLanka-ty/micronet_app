# -*- coding: utf-8 -*-
"""
Created on Sat May 17 23:17:45 2025

@author: BioPixS
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:43:10 2025

Updated version to read interaction matrices from two separate sheets.

@author: BioPixS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from PIL import Image


def load_excel(file):
    interaction1 = pd.read_excel(file, sheet_name='Positive', header=None)
    interaction2 = pd.read_excel(file, sheet_name='Negative', header=None)
    settings_df = pd.read_excel(file, sheet_name='PlotSettings', header=None)
    return interaction1, interaction2, settings_df


def extract_params(settings_df):
    def get_setting(key):
        return settings_df.loc[settings_df[0] == key, 1].values[0]

    return {
        "restfactor": get_setting('restfactor'),
        "linewidth_base": get_setting('linewidth_base'),
        "linewidth_scale": get_setting('linewidth_scale'),
        "arrowsize_base": get_setting('arrowsize_base'),
        "arrowsize_scale": get_setting('arrowsize_scale'),
        "combined_alpha": get_setting('combined_alpha'),
        "plot_min_connectors": get_setting('minconflag'),
        "neutral_strength": get_setting('neutral_strength'),

        "nodeFontSize": int(get_setting('nodeFontSize')) if 'nodeFontSize' in settings_df[0].values else 12
    }


def extract_single_matrix(raw):
    node_names = raw.iloc[0, 1:].tolist()
    N = len(node_names)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            val = raw.iloc[i + 1, j + 1]
            W[i, j] = float(val) if not pd.isna(val) else 0.0
    return node_names, W


def plot_graph(matrix, node_names, filename, edge_color,arrow_color,neutral_color, pos, params):
    matrix = matrix + params["restfactor"] * (np.ones_like(matrix) - np.eye(len(matrix)))
    G = nx.DiGraph()
    for i, src in enumerate(node_names):
        for j, dst in enumerate(node_names):
            if matrix[i, j] != 0:
                G.add_edge(dst, src, weight=matrix[i, j])

    weights = [abs(G[u][v]['weight']) for u, v in G.edges()]
    widths = params["linewidth_base"] + params["linewidth_scale"] * (np.array(weights) / max(weights or [1]))
    arrows = params["arrowsize_base"] + params["arrowsize_scale"] * (np.array(weights) / max(weights or [1]))

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80, node_color='black')
    for node, (x, y) in pos.items():
        ax.text(x, y + 0.06, node, fontsize=params["nodeFontSize"],
                ha='center', va='center', fontstyle='italic',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    min_arrow_size = min(arrows)
    # min_width = min(widths)

    for (u, v), width, arrow_size in zip(G.edges(), widths, arrows):
        if u == v:
            continue

        x_start, y_start = pos[u]
        x_end, y_end = pos[v]
        # Skip edge if not plotting min connectors and this is the minimum size
        if arrow_size == min_arrow_size:
        # Otherwise, draw the edge
            patch = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                    connectionstyle="arc3,rad=0.2",
                                    arrowstyle='-',
                                    color=neutral_color,
                                    linewidth=width*params["neutral_strength"], 
                                    mutation_scale=arrow_size)
            ax.add_patch(patch)        
        else:
            # Otherwise, draw the edge
            patch = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                    connectionstyle="arc3,rad=0.2",
                                    arrowstyle='-',
                                    color=edge_color,
                                    linewidth=width,
                                    mutation_scale=arrow_size)
            ax.add_patch(patch)

                    

        if arrow_size > min_arrow_size:
            rad = 0.2
            ctrl_x = (x_start + x_end) / 2 + rad * (y_end - y_start)
            ctrl_y = (y_start + y_end) / 2 - rad * (x_end - x_start)

            t = 0.5
            mid_x = (1 - t) ** 2 * x_start + 2 * (1 - t) * t * ctrl_x + t ** 2 * x_end
            mid_y = (1 - t) ** 2 * y_start + 2 * (1 - t) * t * ctrl_y + t ** 2 * y_end

            dx = 2 * (1 - t) * (ctrl_x - x_start) + 2 * t * (x_end - ctrl_x)
            dy = 2 * (1 - t) * (ctrl_y - y_start) + 2 * t * (y_end - ctrl_y)
            norm = np.hypot(dx, dy)
            dx, dy = dx / norm * 0.02, dy / norm * 0.02

            rad = 0.1
            mid_arrow = FancyArrowPatch((x_start, y_start), (mid_x + dx, mid_y + dy),
                                        connectionstyle=f"arc3,rad={rad}",
                                        arrowstyle='-|>',
                                        color=arrow_color,
                                        linewidth=0,
                                        mutation_scale=arrow_size * 3)
            ax.add_patch(mid_arrow)

    ax.set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def generate_all_graphs(excel_path):


    edge_colors = {
        "edgeColor1": [110/256, 158/256, 158/256],       # Blue
        "edgeColor2": [245/256, 193/256, 89/256]     # Magenta
    }
    
    arrow_colors = {
        "arrowColor1":[88/256,126/256,139/256],       # Blue
        "arrowColor2": [217/256,	139/256,	75/256]      # Magenta
    }
    neutral_color = [0,0, 0]
    raw1, raw2, settings = load_excel(excel_path)
    params = extract_params(settings)
    node_names1, w1 = extract_single_matrix(raw1)
    node_names2, w2 = extract_single_matrix(raw2)

    
    assert node_names1 == node_names2, "Node names must match between both matrices"
    node_names = node_names1

    G_ref = nx.DiGraph()
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if w1[i, j] + w2[i, j] != 0:
                G_ref.add_edge(node_names[j], node_names[i])
    pos = nx.circular_layout(G_ref)

    plot_graph(w1, node_names, "fig1.png", edge_colors["edgeColor1"],arrow_colors["arrowColor1"],neutral_color, pos, params)
    plot_graph(w2, node_names, "fig2.png", edge_colors["edgeColor2"],arrow_colors["arrowColor2"],neutral_color, pos, params)

    img1 = Image.open("fig1.png").convert("RGBA")
    img2 = Image.open("fig2.png").convert("RGBA")
    blended = Image.blend(img1, img2, params["combined_alpha"])
    blended.convert("RGB").save("combinedFigures.jpg")

    return "fig1.png", "fig2.png", "combinedFigures.jpg"
