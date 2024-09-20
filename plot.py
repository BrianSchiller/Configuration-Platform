from smac import intensifier, runhistory

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from ConfigSpace import Configuration
from types import SimpleNamespace
import os

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


def plot_configs(directories: list[Path]):
    configs = {}
    # Construct config dict by adding one dict per model
    # Also add default model
    for sub_dir in directories[0].iterdir():
        if sub_dir.is_dir() and "B_" in sub_dir.name:
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
    for directory in directories:
        for sub_dir in directory.iterdir():
            if sub_dir.is_dir() and "B_" in sub_dir.name:
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
        df_unsorted['dimension'] = pd.to_numeric(df_unsorted['dimension'])
        df_unsorted['budget'] = pd.to_numeric(df_unsorted['budget'])
        ### general
        # df_unsorted['dimension'] = pd.Categorical(df_unsorted['dimension']).codes
        df = df_unsorted.sort_values(by=['dimension', 'budget']).reset_index(drop=True)
        model_dim = const.MODEL_DIM[model]
        model_cat_att = const.MODEL_CAT_ATT[model]
        model_bin_att = const.MODEL_BIN_ATT[model]

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
            elif column in model_bin_att:
                random_adjustments = np.random.uniform(0, 0.02, size=len(df[column]))
                dim["values"] = np.where(df[column] == 1, df[column] - random_adjustments, df[column] + random_adjustments)
            else:
                dim["values"] = df[column]
            dimensions.append(dim)

        ticktext = []
        for index, row in df.iterrows():
            if row['dimension'] == 0:
                ticktext.append(f"Default")
            else:
                ### General
                # if row['dimension'] == 2:
                #     set_name = "Set: A"
                # elif row['dimension'] == 1:
                #     set_name = "Set: B"
                # else:
                #     set_name = "Set: C"
                # ticktext.append(f"{set_name}; B: {row['budget']}")
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
        
        output = Path("plots") / directories[0].name
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
    
def plot_test_performance(dir: Path, dims: list[int] = None, buds: list[int] = None, line: bool = False, heat: bool = False, violin: bool = False):
    if dims == None:
        dims = [[2], [3], [5], [10], [15]]
    if buds == None:
        buds = [200, 300, 500, 999, 1500]

    results = {}
    violin_results = {}
    threshold = 10**(-10)
    for dim in dims:
        for bud in buds:
            bud_dim_dir = Path(os.path.join(dir, f"B_{bud}__D_{'_'.join(map(str, dim))}"))
        
            # Iterate over model folders
            model_dirs = [f for f in bud_dim_dir.iterdir() if f.is_dir()]
            for model_dir in model_dirs:
                
                if not model_dir.name in results:
                    results[model_dir.name] = {}
                    violin_results[model_dir.name] = {}
                if not '_'.join(map(str, dim)) in results[model_dir.name]:
                    results[model_dir.name]['_'.join(map(str, dim))] = {}
                    violin_results[model_dir.name]['_'.join(map(str, dim))] = {}

                test_results = model_dir / "test_results"
                top_results = test_results / "top_results"
                default_results = test_results / "default_results"

                top_csvs = [csv for csv in top_results.iterdir() if csv.is_file()]
                top_res = []
                top_vals = []
                for csv in top_csvs:
                    df_top = pd.read_csv(csv)
                    df_top['log_loss'] = np.where(df_top['loss'] < threshold, -10, np.log10(df_top['loss']))
                    avg_loss = df_top['log_loss'].mean()
                    top_res.append(avg_loss)
                    top_vals.append(df_top)
                
                def_csvs = [csv for csv in default_results.iterdir() if csv.is_file()]
                def_res = []
                def_vals = []
                for csv in def_csvs:
                    df_def = pd.read_csv(csv)
                    df_def['log_loss'] = np.where(df_def['loss'] < threshold, -10, np.log10(df_def['loss']))
                    avg_loss = df_def['log_loss'].mean()
                    def_res.append(avg_loss)
                    def_vals.append(df_def)

                results[model_dir.name]['_'.join(map(str, dim))][bud] = {
                    "top": top_res,
                    "def": def_res
                }
                violin_results[model_dir.name]['_'.join(map(str, dim))][bud] = {
                    "top": pd.concat(top_vals),
                    "def": pd.concat(def_vals)
                }

    output_dir = Path(os.path.join(dir, "test_performance"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if line:
        output = output_dir / "line" / f"dim_{'_'.join(map(str, dim))}"
        output.mkdir(parents=True, exist_ok=True)
        plot_performance_line(output, results)
    if heat:
        output = output_dir / "heat" 
        output.mkdir(parents=True, exist_ok=True)
        plot_performance_heat(output, results)
    if violin:
        return violin_results
        # output = output_dir / "violin" 
        # output.mkdir(parents=True, exist_ok=True)
        # plot_violin(output, violin_results, results)

def categorize_problem(problem):
                    if problem <= 5:
                        return 'separable'
                    elif problem <= 9:
                        return 'low/moderate conditioning'
                    elif problem <= 14:
                        return 'strong conditioning, unimodal'
                    elif problem <= 19:
                        return 'multi-modal, adequate global structure'
                    else:
                        return 'multi-modal, weak global structure'

def plot_violin(output: Path, wizards = False, basic = False):
    problem_palette = {
                    'separable': '#b0b0b0',  
                    'low/moderate conditioning': '#708c8c',
                    'strong conditioning, unimodal': '#696969',
                    'multi-modal, adequate global structure': '#2F4F4F',  
                    'multi-modal, weak global structure': '#343434' 
                }
    
    if basic:
        specialised = plot_test_performance(Path("Output/Final"), violin=True)
        basic_diff = []
        for model_name, model_data in specialised.items(): 
            for dimension, dimension_data in model_data.items():
                for budget, budget_data in dimension_data.items():
                    top_df = budget_data['top']
                    def_df = budget_data['def']
                    
                    # Compute log_loss differences
                    diff = top_df['log_loss'] - def_df['log_loss']
                    problem = top_df["problem"]
                    diff_df = pd.DataFrame({
                        'model': model_name,
                        'dimension': int(dimension),
                        'budget': budget,
                        "problem": problem,
                        'log_loss_diff': diff,
                        "log_loss": top_df["log_loss"],
                        "name": "Specialised"
                    })
                    basic_diff.append(diff_df)
        df_spec = pd.concat(basic_diff)

        dimensions = sorted(df_spec['dimension'].unique())
        budgets = sorted(df_spec['budget'].unique())
        
        for budget in budgets:
            for dim in dimensions:
                subset_spec = df_spec[(df_spec['budget'] == budget) & (df_spec['dimension'] == dim)]

                subset_spec['bbob problem'] = subset_spec['problem'].apply(categorize_problem)

                mean_log_loss_diff = subset_spec.groupby('model')['log_loss'].mean().reset_index()
                palette = {
                    'CMA': '#4381e6',
                    'MetaModel': '#fcef74',
                    'MetaModelFmin2': '#cc7ff0',
                    'MetaModelOnePlusOne': '#e05164',
                    'ChainMetaModelPowell': '#4ecf73'
                }

                # Plotting the violin plot
                plt.figure(figsize=(12, 8))
                sns.violinplot(x='model', y='log_loss', data=subset_spec, inner=None, palette=palette, hue="model", legend=False)
                sns.stripplot(x='model', y='log_loss', data=subset_spec, jitter=True, dodge=True, alpha=0.7, hue='bbob problem', palette=problem_palette)
                for i, name in enumerate(mean_log_loss_diff['model']):
                    plt.hlines(y=mean_log_loss_diff.loc[i, 'log_loss'], xmin=i-0.2, xmax=i+0.2, color='red')
                plt.xticks(rotation=45)
                plt.title(f'Log Loss Performance for dimension {dim}, budget {budget}')
                plt.ylabel('Log Loss Performance')
                plt.xlabel('Optimiser')
                plt.tight_layout()
                output_dir = output / "basic" 
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{dim}_{budget}.pdf"
                plt.savefig(output_file)
                print(output_file)
    
    if wizards:
        specialised = plot_test_performance(Path("Output/Final"), buds=[200,300,500], violin=True)
        scenarioAB = plot_test_performance(Path("Output/Multi_Dim Scenario AB"), 
                                        dims=[[2,3,5],[10,15]], buds=[200,300,500], violin=True)
        scenarioC = plot_test_performance(Path("Output/Multi_Dim_2351015"),
                                        dims=[[2,3,5,10,15]], buds=[200,300,500], violin=True)
        
        specialised_diff = []
        for model_name, model_data in specialised.items(): 
            for dimension, dimension_data in model_data.items():
                for budget, budget_data in dimension_data.items():
                    top_df = budget_data['top']
                    def_df = budget_data['def']
                    
                    # Compute log_loss differences
                    diff = top_df['log_loss'] - def_df['log_loss']
                    problem = top_df["problem"]
                    diff_df = pd.DataFrame({
                        'model': model_name,
                        'dimension': int(dimension),
                        'budget': budget,
                        "problem": problem,
                        'log_loss_diff': diff,
                        'log_loss': top_df["log_loss"],
                        "name": "Specialised"
                    })
                    specialised_diff.append(diff_df)
        df_spec = pd.concat(specialised_diff)

        ab_diff = []
        for model_name, model_data in scenarioAB.items(): 
            for dimension, dimension_data in model_data.items():
                for budget, budget_data in dimension_data.items():
                    top_df = budget_data['top']
                    def_df = budget_data['def']

                    dimensions = top_df["dimension"].unique()

                    for dim in dimensions:
                        top_sub = top_df[(top_df["dimension"]) == dim].reset_index()
                        def_sub = def_df[(top_df["dimension"]) == dim].reset_index()
                        diff = top_sub['log_loss'] - def_sub['log_loss']
                        problem = top_sub["problem"]
                        diff_df = pd.DataFrame({
                            'model': model_name,
                            'dimension': dim,
                            'budget': budget,
                            "problem": problem,
                            'log_loss_diff': diff,
                            'log_loss': top_sub["log_loss"],
                            'name': "Scenario A/B"
                        })
                        ab_diff.append(diff_df)
        df_ab = pd.concat(ab_diff)

        c_diff = []
        for model_name, model_data in scenarioC.items(): 
            for dimension, dimension_data in model_data.items():
                for budget, budget_data in dimension_data.items():
                    top_df = budget_data['top']
                    def_df = budget_data['def']

                    dimensions = top_df["dimension"].unique()

                    for dim in dimensions:
                        top_sub = top_df[(top_df["dimension"]) == dim].reset_index()
                        def_sub = def_df[(top_df["dimension"]) == dim].reset_index()
                        diff = top_sub['log_loss'] - def_sub['log_loss']
                        problem = top_sub["problem"]
                        diff_df = pd.DataFrame({
                            'model': model_name,
                            'dimension': dim,
                            'budget': budget,
                            "problem": problem,
                            'log_loss_diff': diff,
                            'log_loss': top_sub["log_loss"],
                            "name": "Scenario C"
                        })
                        c_diff.append(diff_df)
        df_c = pd.concat(c_diff)

        dimensions = sorted(df_spec['dimension'].unique())
        budgets = sorted(df_spec['budget'].unique())
        models = sorted(df_spec['model'].unique())
        
        for model in models:
            for budget in budgets:
                for dim in dimensions:
                    subset_spec = df_spec[(df_spec['model'] == model) & (df_spec['budget'] == budget) & (df_spec['dimension'] == dim)]
                    subset_ab = df_ab[(df_ab['model'] == model) & (df_ab['budget'] == budget) & (df_ab['dimension'] == dim)]
                    subset_c = df_c[(df_c['model'] == model) & (df_c['budget'] == budget) & (df_c['dimension'] == dim)]

                    df_diff = pd.concat([subset_spec, subset_ab, subset_c])
                    df_diff['bbob problem'] = df_diff['problem'].apply(categorize_problem)

                    mean_log_loss_diff = df_diff.groupby('name')['log_loss'].mean().reset_index()

                    # Plotting the violin plot
                    plt.figure(figsize=(12, 8))
                    sns.violinplot(x='name', y='log_loss', data=df_diff, inner=None, color="lightsalmon")
                    sns.stripplot(x='name', y='log_loss', data=df_diff, jitter=True, dodge=True, alpha=0.7, hue='bbob problem', palette=problem_palette)
                    for i, name in enumerate(mean_log_loss_diff['name']):
                        plt.hlines(y=mean_log_loss_diff.loc[i, 'log_loss'], xmin=i-0.2, xmax=i+0.2, color='red')
                    plt.xticks(rotation=45)
                    plt.title(f'Log Loss Performance for dimension {dim}, budget {budget}')
                    plt.ylabel('Log Loss Performance')
                    plt.xlabel('Wizard')
                    plt.tight_layout()
                    output_dir = output / "wizards" / f"{dim}_{budget}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"{model}.pdf"
                    plt.savefig(output_file)
                    print(output_file)
    

def plot_performance_line(output_dir: Path, results: dict):
    # Colors
    def generate_color_map(n_colors):
        return cm.get_cmap('viridis', n_colors)
    unique_combinations = set()
    for dim_dict in results.values():
        for dim, bud_dict in dim_dict.items():
            for bud, dict in bud_dict.items():
                unique_combinations.add((dim, bud))  
    unique_combinations = sorted(unique_combinations)  
    color_map = generate_color_map(len(unique_combinations))
    combination_to_color = {}
    for idx, (dim, bud) in enumerate(unique_combinations):
        combination_to_color[(dim, bud)] = color_map(idx)

    # Legend
    configured_handle = Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Configured')
    default_handle = Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Default')

    for model, dim_dict in results.items():
        model_file_path = output_dir / f"{model}.pdf"
        handles = [configured_handle, default_handle]

        plt.clf()
        for dim, bud_dict in dim_dict.items():
            for bud, res_dict in bud_dict.items():
                color = combination_to_color[(dim, bud)]
                plt.plot(range(1, len(res_dict["top"]) + 1), res_dict["top"], color=color, linestyle='-', marker="o", label=f"Dimension: {dim}, Budget: {bud}")
                plt.plot(range(1, len(res_dict["def"]) + 1), res_dict["def"], color=color, linestyle='--', marker="o")
                handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=2, label=f"Dimension: {dim.replace('_', ',')}, Budget: {bud}"))
        plt.legend(handles = handles, loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.gcf().set_size_inches(12, 6)  # Adjusted size for better visibility
        plt.tight_layout(pad=3.0)
        plt.xlabel('Evaluation')
        plt.ylabel('Fitness (Log Loss)')
        plt.title('Performance Evaluation Results')
        plt.xticks(range(1, len(res_dict["top"]) + 1))
        plt.savefig(model_file_path)


def plot_performance_heat(output_dir: Path, results: dict):
    models = list(results.keys())
    dims = list(results[models[0]].keys())
    buds =list(results[models[0]][dims[0]].keys())

    for model in models:
        heatmap_data = np.zeros((len(dims), len(buds)))
        annotations = np.empty((len(dims), len(buds)), dtype=object)
        data = results[model]

        for i, x in enumerate(dims):
            for j, y in enumerate(buds):
                conf = data[x][y]["top"]
                default = data[x][y]["def"]
                heatmap_data[i, j] = round((np.mean(conf) - np.mean(default)), 2)
                annotations[i, j] = f'{round((np.mean(conf) - np.mean(default)), 2)} ({round(np.mean(conf), 2)})'

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, xticklabels=buds, yticklabels=dims, annot=annotations, fmt='', cmap='viridis')
        plt.xlabel('Budget')
        plt.ylabel('Dimension')
        plt.title('Log Loss Performance Improvement (Overall Performance)')
        model_file_path = output_dir / f"{model}.pdf"
        plt.savefig(model_file_path)
        print(f"Plotted Heatmap for {model}: {model_file_path}")


if __name__ == "__main__":
    dir = Path("Output/Final")
    # dirs = [Path("Output/Multi_Dim_235"), Path("Output/Multi_Dim_1015") ,Path("Output/Multi_Dim_2351015")]
    # plot_configs([dir])
    # dims = [[2, 3, 5, 10, 15]]
    # for dim in dims:
    plot_test_performance(dir, heat = True)
    # plot_violin(Path("plots/violin"), wizards=True)