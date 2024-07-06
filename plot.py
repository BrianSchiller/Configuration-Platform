from smac import intensifier

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
from pathlib import Path
from ConfigSpace import Configuration
import plotly.express as px

from models import MetaModelOnePlusOne
import constants as const
import settings

def plot_config_difference(config_path: Path, default_config, model_name: str, output: Path): 
    # Recreate configuration
    with open(config_path, 'r') as file:
        config_string = file.read()
    config_dict = eval(config_string)
    config = Configuration(default_config.config_space, config_dict)

    attributes = config.keys()
    df = pd.DataFrame([config.get_array(), default_config.get_array()], columns=attributes, index=['Optimised', 'Default'])
    df = df.reset_index()

    plt.figure(figsize=(12,6))  # Adjust figure size if necessary
    parallel_coordinates(df, 'index', color=['red', 'green'])
    plt.title('Parallel Coordinate Plot')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.xticks(rotation=45, ha='right')  # Rotate and align x-axis labels to the right
    plt.tight_layout()
    plt.savefig(output / 'config_difference.png')


def plot_trajectory(intensifier: intensifier, unique_directory: Path, model_name: str):
    incumbents = []
    for index in range(0, len(intensifier.trajectory)):
        config = intensifier.trajectory[index]
        if index != len(intensifier.trajectory) - 1:
            end = intensifier.trajectory[index + 1].trial - 1
            incumbents.append((config.config_ids[0], config.trial, end))
        else:
            incumbents.append((config.config_ids[0], config.trial, settings.trials))

    trajectory_df = pd.DataFrame(incumbents, columns=['Configuration', 'Start', 'End'])

    # Plotting the timeline
    plt.figure(figsize=(12, 6))

    config_positions = {config: i for i, config in enumerate(trajectory_df['Configuration'])}
    for i, row in trajectory_df.iterrows():
        y_pos = config_positions[row['Configuration']]
        plt.hlines(y=y_pos, xmin=row['Start'], xmax=row['End'], linewidth=5, label=row['Configuration'])

    plt.xlabel('Iteration')
    plt.ylabel('Configuration')
    plt.title('Timeline of Best Configurations')
    plt.yticks(ticks=list(config_positions.values()), labels=list(config_positions.keys()))
    plt.gca().invert_yaxis()
    plt.grid(True)
    output_file = unique_directory / f"{model_name}.png"
    plt.savefig(output_file)
    print(f"Plotted {model_name} trajectory: {output_file}")


if __name__ == "__main__":
    config = Path("Output/MetaModelOnePlusOne-Test-20240703_11-01-41/B_200__D_5/MetaModelOnePlusOne/MetaModelOnePlusOne_B_200_D_5.txt")
    model = MetaModelOnePlusOne(None)
    default_config = model.configspace.get_default_configuration()
    output = Path("Output/MetaModelOnePlusOne-Test-20240703_11-01-41/B_200__D_5/MetaModelOnePlusOne")

    plot_config_difference(config, default_config, "MetaModelOnePlusOne", output)