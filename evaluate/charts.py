import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from logging_config import logger


def plot_finetuning_metrics(experiment_root, param_folders, param_values, parameter):
    # Define the metrics to extract
    metrics_to_plot = ['ROUGE_L', 'Exact Match', 'METEOR']

    # Prepare a figure with subplots (one row, multiple columns)
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * 5, 5))

    if num_metrics == 1:
        axes = [axes]  # Ensure axes is always a list for iteration

    metric_data = {metric: {} for metric in metrics_to_plot}  # Store metric values for each model
    model_names = set()

    for param_folder, param_value in zip(param_folders, param_values):
        folder_path = os.path.join(experiment_root, param_folder)

        # Iterate through metric files in the experiment folder
        if not os.path.exists(folder_path):
            logger.info(f"Warning: Folder {folder_path} does not exist.")
            continue

        metrics_file = os.path.join(folder_path, 'metrics.json')

        if not os.path.exists(metrics_file):
            continue

        with open(metrics_file, 'r') as f:
            data = json.load(f)

            # Extract relevant metric values from first three dictionaries
            for entry in data[:3]:  # Only consider the first 3 dicts
                for model_name, values in entry.items():
                    model_names.add(model_name)

                    for metric in metrics_to_plot:
                        if model_name not in metric_data[metric]:
                            metric_data[metric][model_name] = []
                        metric_value = values['ROUGE']['rougeL'] if metric == 'ROUGE_L' else values.get(metric,
                                                                                                        None)
                        if metric_value is not None:
                            metric_data[metric][model_name].append((param_value, metric_value))

    # Generate scatter plots for each metric
    for col_idx, metric in enumerate(metrics_to_plot):
        ax = axes[col_idx]

        for model_name in model_names:
            if model_name in metric_data[metric] and metric_data[metric][model_name]:
                param_values_sorted, metric_values_sorted = zip(*sorted(metric_data[metric][model_name]))
                ax.scatter(param_values_sorted, metric_values_sorted, label=model_name)
                ax.plot(param_values_sorted, metric_values_sorted, linestyle='-', alpha=0.7)

        ax.set_title(f"{metric} vs {parameter}", fontsize=16)
        ax.set_xlabel(f"{parameter}", fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.set_ylim(0, max(max(val for _, val in values) for values in metric_data[metric].values() if values) * 1.1)
        ax.legend(loc='best', fontsize=13, framealpha=0.5, alignment='left')
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

    plt.tight_layout()

    plt.savefig(f'experiments/charts/{parameter}', dpi=300)

    return None
