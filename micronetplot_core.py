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
        "scale neutral strength": get_setting('scale neutral strength'),
        "scale linewidth": get_setting('scale linewidth'),
        "scale arrowsize": get_setting('scale arrowsize'),
        "combined transparency": get_setting('combined transparency'),
        "nodeSize" :  int(get_setting('nodeSize')) if 'nodeSize' in settings_df[0].values else 8,
        "nodeFontSize": int(get_setting('nodeFontSize')) if 'nodeFontSize' in settings_df[0].values else 12,
        "nodeColor": get_setting('nodeColor').strip(),
        "Positive Line Color": get_setting('Positive Line Color').strip(),
        "Positive Arrow Color": get_setting('Positive Arrow Color').strip(),
        "Negative Line Color": get_setting('Negative Line Color').strip(),
        "Negative Arrow Color": get_setting('Negative Arrow Color').strip(),
        "Neutral Color": get_setting('Neutral Color').strip(),
        
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


def plot_graph(matrix, node_names, filename, pos, params, inttype):
    baseline_factor = 0.0001 # Strength of the 0 values in the Interaction Matrix
    baseline_width = 0.1
    baseline_arrowsize = 4
    if inttype == 1:
        edge_color = params["Positive Line Color"] 
        arrow_color = params["Positive Arrow Color"] 
        neutral_color = params["Neutral Color"] 
    else:
        edge_color = params["Negative Line Color"] 
        arrow_color = params["Negative Arrow Color"] 
        neutral_color = params["Neutral Color"] 
        
    matrix = matrix + baseline_factor * (np.ones_like(matrix) - np.eye(len(matrix)))
    G = nx.DiGraph()
    for i, src in enumerate(node_names):
        for j, dst in enumerate(node_names):
            if matrix[i, j] != 0:
                G.add_edge(dst, src, weight=matrix[i, j])

    weights = [abs(G[u][v]['weight']) for u, v in G.edges()]
    widths = baseline_width + params["scale linewidth"] * (np.array(weights) / max(weights or [1])) # 
    arrows = baseline_arrowsize +  params["scale arrowsize"] * (np.array(weights) / max(weights or [1])) # 

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=params["nodeSize"], node_color = params["nodeColor"] ) # 
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
                                    linewidth=width*params["scale neutral strength"], 
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

    plot_graph(w1, node_names, "fig1.png", pos, params, 1)
    plot_graph(w2, node_names, "fig2.png", pos, params, 2)

    img1 = Image.open("fig1.png").convert("RGBA")
    img2 = Image.open("fig2.png").convert("RGBA")
    blended = Image.blend(img1, img2, params["combined transparency"])
    blended.convert("RGB").save("combinedFigures.jpg")

    return "fig1.png", "fig2.png", "combinedFigures.jpg"
