import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def boxplot_results(results, parameter_name, filename=None):
    methods = sorted(results.keys())
    param_values = sorted(list(results[methods[0]].keys()))

    records = []
    for method in methods:
        for val in param_values:
            for acc in results[method][val]:
                records.append([method, val, acc])

    df = pd.DataFrame(records, columns=["Method", parameter_name, "Accuracy"])
    plt.figure()

    unique_vals = param_values
    n_vals = len(unique_vals)
    method_list = ["LDA", "QDA", "NB"]
    offset_map = {"LDA": -0.2, "QDA": 0.0, "NB": 0.2}

    def get_data(m, v):
        return df[(df["Method"] == m) & (df[parameter_name] == v)]["Accuracy"].values

    fig, ax = plt.subplots()

    x_positions = np.arange(n_vals)

    box_colors = {'LDA': 'tan', 'QDA': 'lightblue', 'NB': 'lightgreen'}  # or any color scheme

    for i, val in enumerate(unique_vals):
        for m in method_list:
            data_m_v = get_data(m, val)
            pos = x_positions[i] + offset_map[m]
            bp = ax.boxplot(data_m_v, positions=[pos], widths=0.15, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(box_colors[m])

            for flier in bp['fliers']:
                flier.set(marker='o', markersize=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{v}" for v in unique_vals])
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Scheme - {parameter_name} variation")

    # Legend
    legend_elements = [
        Patch(facecolor=box_colors['LDA'], label='LDA'),
        Patch(facecolor=box_colors['QDA'], label='QDA'),
        Patch(facecolor=box_colors['NB'], label='NB')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close()


