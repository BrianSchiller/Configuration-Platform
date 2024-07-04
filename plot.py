import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
from pathlib import Path
from ConfigSpace import Configuration
import plotly.express as px

from models import MetaModelOnePlusOne
import constants as const

def plot_config_difference(config_path: Path, default_config, model_name: str, output: Path): 
    # Recreate configuration
    with open(config_path, 'r') as file:
        config_string = file.read()
    confic_dict = eval(config_string)
    config = Configuration(default_config.config_space, confic_dict)

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




if __name__ == "__main__":
    config = Path("Output/MetaModelOnePlusOne-Test-20240703_11-01-41/B_200__D_5/MetaModelOnePlusOne/MetaModelOnePlusOne_B_200_D_5.txt")
    model = MetaModelOnePlusOne(None)
    default_config = model.configspace.get_default_configuration()
    output = Path("Output/MetaModelOnePlusOne-Test-20240703_11-01-41/B_200__D_5/MetaModelOnePlusOne")

    plot_config_difference(config, default_config, "MetaModelOnePlusOne", output)