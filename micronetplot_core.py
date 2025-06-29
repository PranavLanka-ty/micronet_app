# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 13:53:53 2025

@author: Pranav Lanka
"""

"""
Updated: Generates positive and negative interaction graphs with transparency masks for precise blending.
"""

# micronetplot_core.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from PIL import Image

def plot_graph(W_pos, W_neg, node_names, filename, pos, params,
               highlight='positive', hide_neutral=False, hide_nodes=False, hide_labels=False,transp_flag = False):
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
    linewidths = (params["Linewidth"]/100) + params["Linewidth"] * (weights / max_weight)
    arrowsizes = baseline['arrow'] + params["Arrowsize"] * (weights / max_weight)

    fig, ax = plt.subplots(figsize=(10, 10))

    if hide_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_size=params["nodeSize"],
                               node_color=params(1, 1, 1, 0))
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_size=params["NodeSize"],
                               node_color=params["NodeColor"])
            
    if hide_labels:
        for node, (x, y) in pos.items():
            ax.text(x, y + 0.06, node, fontsize=params["FontSize"],
                    ha='center', va='center', fontstyle='italic', color = (1, 1, 1, 0),
                    bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2'))
    else:
        for node, (x, y) in pos.items():
            ax.text(x, y + 0.06, node, fontsize=params["FontSize"],
                    ha='center', va='center', fontstyle='italic',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    for (u, v), lw, arrow_size in zip(G.edges(), linewidths, arrowsizes):
        if u == v:
            continue
        i, j = node_names.index(v), node_names.index(u)

        is_positive = W_pos[i, j] != 0
        is_negative = W_neg[i, j] != 0
        is_neutral = not is_positive and not is_negative


        if is_positive:
            edge_color = params["Positive Line Color"]
            arrow_color = params["Positive Arrow Color"]
        elif is_negative:
            edge_color = params["Negative Line Color"]
            arrow_color = params["Negative Arrow Color"]
        else:
            edge_color = params["Neutral Color"]
            arrow_color =  (1, 1, 1, 0)
            if hide_neutral and is_neutral:
                edge_color = (1, 1, 1, 0)
                arrow_color =  (1, 1, 1, 0)


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

        t = 0.5
        mid_x = (1 - t)**2 * x_start + 2 * (1 - t) * t * ctrl_x + t**2 * x_end
        mid_y = (1 - t)**2 * y_start + 2 * (1 - t) * t * ctrl_y + t**2 * y_end
        dx = 2 * (1 - t) * (ctrl_x - x_start) + 2 * t * (x_end - ctrl_x)
        dy = 2 * (1 - t) * (ctrl_y - y_start) + 2 * t * (y_end - ctrl_y)
        norm = np.hypot(dx, dy)
        dx, dy = (dx / norm, dy / norm) if norm != 0 else (1.0, 0.0)

        arrow = FancyArrowPatch((mid_x - dx * 0.01, mid_y - dy * 0.01),
                                (mid_x + dx * 0.01, mid_y + dy * 0.01),
                                connectionstyle="arc3,rad=0",
                                arrowstyle='-|>',
                                color=arrow_color,
                                linewidth=0,
                                mutation_scale=arrow_size * 3,
                                zorder=10)
        ax.add_patch(arrow)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    
    if transp_flag is True:
        plt.savefig(filename, transparent=True)#, transparent=True
    else:
        plt.savefig(filename)#, transparent=True
        
    plt.close()
    

def generate_all_graphs(excel_path):
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
            "Linewidth": get('Linewidth', 1.0, float),
            "Arrowsize": get('Arrowsize', 1.0, float),
            "NodeSize": get('NodeSize', 8, int),
            "FontSize": get('FontSize', 12, int),
            "NodeColor": get('NodeColor', 'skyblue', str),
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

    df_pos, df_neg, df_settings = load_excel(excel_path)
    params = extract_settings(df_settings)
    node_names, W_pos = extract_matrix(df_pos)
    _, W_neg = extract_matrix(df_neg)

    G_all = nx.DiGraph()
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if W_pos[i, j] + W_neg[i, j] != 0:
                G_all.add_edge(node_names[j], node_names[i])
    pos = nx.circular_layout(G_all)

    # Full versions
    plot_graph(W_pos, W_neg, node_names, "fig1.png", pos, params, highlight='positive', transp_flag = False)
    plot_graph(W_pos, W_neg, node_names, "fig2.png", pos, params, highlight='negative', transp_flag = False)

    # Dummy fig1 for clean overlay
    plot_graph(W_pos, W_neg, node_names, "fig1_dummy.png", pos, params,
                highlight='positive', hide_neutral=True, hide_nodes=False, hide_labels=False, transp_flag = True)

    # Dummy fig2 for clean overlay
    plot_graph(W_pos, W_neg, node_names, "fig2_dummy.png", pos, params,
                highlight='negative', hide_neutral=False, hide_nodes=False, hide_labels=True, transp_flag = True)


    img1 = Image.open("fig1_dummy.png").convert("RGBA")
    img2 = Image.open("fig2_dummy.png").convert("RGBA")
    
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    combined = Image.alpha_composite(img2, img1)
    
    # # Force RGB (drop alpha channel)
    # combined_rgb = combined.convert("RGB")
    # combined_rgb.save("combinedFigures.png", dpi=(600, 600))
    
    # Upscale smoothly
    scale_factor = 3  # 3x upscaling
    new_size = (combined.width * scale_factor, combined.height * scale_factor)
    combined_upscaled = combined.resize(new_size, resample=Image.Resampling.LANCZOS)

    # Properly remove transparency by pasting on a white background
    background = Image.new("RGB", combined_upscaled.size, (255, 255, 255))  # white background
    background.paste(combined_upscaled, mask=combined_upscaled.split()[3])  # paste using alpha channel as mask
    
    # Save final image with no transparency
    background.save("combinedFigures.png", dpi=(300, 300))    
    
    return "fig1.png", "fig2.png", "fig2_dummy.png", "combinedFigures.png"

