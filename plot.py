from smac import intensifier, runhistory

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from ConfigSpace import Configuration
import plotly.express as px
import json
from types import SimpleNamespace
import os

from models import MetaModelOnePlusOne
import constants as const
import settings

def plot_config_difference(config_path: Path, default_config, model_name: str, output: Path): 
    # Recreate configuration
    with open(config_path, 'r') as file:
        config_string = file.read()
    config_dict = eval(config_string)
    default_config = dict(default_config)

    df = pd.DataFrame([default_config, config_dict])
    model_dim = const.MODEL_DIM[model_name]
    model_cat_att = const.MODEL_CAT_ATT[model_name]

    for column in model_cat_att:
        categories = df[column].unique()
        df[column] = pd.Categorical(df[column], categories=categories)

    dimensions = []
    for column in df.columns:
        dim = model_dim[column]
        if column in model_cat_att:
            dim["values"] = df[column].cat.codes
            dim["ticktext"] = df[column].unique()
            dim["tickvals"] = list(range(len(dim["ticktext"])))
        else:
            dim["values"] = df[column]
        dimensions.append(dim)
    
    # Create Plotly Parcoords plot
    fig = go.Figure(data=go.Parcoords(line=dict(color=df.index,
                                            colorscale='Viridis',
                                            colorbar=dict(title='Color Scale', tickvals=[0, 1], ticktext=['default', 'configured']),
                                            showscale=True,
                                            cmin=0,
                                            cmax=len(df)-1),
                                    dimensions=dimensions))

    fig.update_layout(title='Parallel Coordinates Plot',
                    plot_bgcolor='white',
                    paper_bgcolor='white')
    
    os.makedirs(output, exist_ok=True)
    output_path = output / 'config_difference.pdf'
    pio.write_image(fig, output_path)
    print(f"Overview over config difference of {model_name}: {output_path}")
    print()


def plot_configs(directory: Path):
    configs = {}
    # Construct config dict by adding one dict per model
    # Also add default model
    for sub_dir in directory.iterdir():
        if sub_dir.is_dir():
            for model in sub_dir.iterdir():
                if model.is_dir():
                    configs[model.name] = {}
                    with open(f"default/{model.name}.txt", 'r') as file:
                        config_string = file.read()
                    config_dict = eval(config_string)
                    config_dict["budget"] = 0
                    config_dict["dimension"] = 0
                    configs[model.name]["default"] = config_dict
                    
            break
    # Read configs for each dim x budget combination
    for sub_dir in directory.iterdir():
        if sub_dir.is_dir():
            for model in sub_dir.iterdir():
                if model.is_dir():
                    with open(model / f"{model.name}_{sub_dir.name.replace('__', '_')}.txt", 'r') as file:
                        config_string = file.read()
                    config_dict = eval(config_string)
                    # Add dim and budget
                    parts = sub_dir.name.split('__')
                    config_dict["budget"] = parts[0][2:]
                    config_dict["dimension"] = parts[1][2:]
                    configs[model.name][sub_dir.name] = config_dict

    for model in configs:
        df_unsorted = pd.DataFrame([config for config in configs[model].values()])
        df = df_unsorted.sort_values(by=['dimension', 'budget']).reset_index(drop=True)
        model_dim = const.MODEL_DIM[model]
        model_cat_att = const.MODEL_CAT_ATT[model]

        for column in model_cat_att:
            categories = df[column].unique()
            df[column] = pd.Categorical(df[column], categories=categories)
        
        dimensions = []
        for column in df.columns:
            dim = model_dim[column]
            if column in model_cat_att:
                dim["values"] = df[column].cat.codes
                dim["ticktext"] = df[column].unique()
                dim["tickvals"] = list(range(len( dim["ticktext"])))
            else:
                dim["values"] = df[column]
            dimensions.append(dim)

        ticktext = []
        for index, row in df.iterrows():
            ticktext.append(f"D: {row['dimension']}, B: {row['budget']}")

         # Create Plotly Parcoords plot
        fig = go.Figure(data=go.Parcoords(line=dict(color=df.index,
                                                colorscale='Turbo',
                                                colorbar=dict(title='Configurations', tickvals=df.index, ticktext=ticktext),
                                                showscale=True,
                                                cmin=0,
                                                cmax=len(df)-1),
                                        dimensions=dimensions))

        fig.update_layout(title='Parallel Coordinates Plot',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        width=1000,
                        height=600)
        
        output = Path("plots") / directory.name
        os.makedirs(output, exist_ok=True)
        pio.write_image(fig, f"{output}/{model}.pdf")
        print(f"Overview over configurations of {model}: {output}/{model}.pdf")
        print()


def plot_trajectory(intensifier: intensifier, runhistory: runhistory, unique_directory: Path, model_name: str):
    incumbents = []
    for index in range(0, len(intensifier.trajectory)):
        config = intensifier.trajectory[index]
        if index != len(intensifier.trajectory) - 1:
            end = intensifier.trajectory[index + 1].trial - 1
            incumbents.append((config.config_ids[0], config.trial, end))
        else:
            incumbents.append((config.config_ids[0], config.trial, settings.trials))
    trajectory_df = pd.DataFrame(incumbents, columns=['Configuration', 'Start', 'End'])

    # Collect real start time to show begin of training
    config_id_start = {}
    for idx, data in enumerate(runhistory._data):
        if data.config_id not in config_id_start:
            config_id_start[data.config_id] = idx + 1

    training = []
    for _, config in trajectory_df.iterrows():
        training.append((config["Configuration"], config_id_start[config["Configuration"]], config["Start"]))
    training_df = pd.DataFrame(training, columns=['Configuration', 'Start', 'End'])

    # Plotting the timeline
    plt.figure(figsize=(12, 6))

    config_positions = {config: i for i, config in enumerate(trajectory_df['Configuration'])}
    for i, row in training_df.iterrows():
        y_pos = config_positions[row['Configuration']]
        plt.hlines(y=y_pos, xmin=row['Start'], xmax=row['End'], linewidth=5, label=row['Configuration'], colors="gold")
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

def dict_to_simplenamespace(d):
    """
    Recursively converts a dictionary to a SimpleNamespace, including nested dictionaries.
    """
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_simplenamespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_simplenamespace(i) for i in d]
    else:
        return d

if __name__ == "__main__":
    dir = Path("Output/_Final-D1015-B235")
    plot_configs(dir)