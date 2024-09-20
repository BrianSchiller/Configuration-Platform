import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import numpy as np

from scipy.stats import ttest_rel


def folder_to_csv(path: Path):
    df_list = []

    # Iterate over all files in the folder
    for filename in os.listdir(path):
        # Check if the file is a CSV
        if filename.endswith('.csv') and filename.startswith("0_"):
            # Construct the full file path
            file_path = os.path.join(path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Append the DataFrame to the list
            df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def analyse_results(path: Path):

    # Load the CSV files into dataframes
    top_df = folder_to_csv(path / "test_results" / "top_results")
    default_df = folder_to_csv(path / "test_results" / "default_results")
    # random_df = folder_to_csv(path / "test_results" / "sample_results")

    # Add a column indicating the configuration type
    top_df['configuration'] = 'Top'
    default_df['configuration'] = 'Default'
    # random_df['configuration'] = 'Random'

    # Concatenate all dataframes into one
    combined_df = pd.concat([top_df, default_df], ignore_index=True)

    # print(stats)

    combined_df['log_loss'] = np.where(combined_df['loss'] < 10**(-10), -10, np.log10(combined_df['loss']))
    top_losses = combined_df[combined_df['configuration'] == 'Top']['log_loss']
    default_losses = combined_df[combined_df['configuration'] == 'Default']['log_loss']

    # Perform paired t-test
    stats = combined_df.groupby('configuration')['log_loss'].agg(['mean', 'median', 'std'])
    t_stat, p_value = ttest_rel(top_losses, default_losses)
    # print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value}")

    stats["p-value"] = p_value    

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='configuration', y='loss', data=combined_df)
    plt.yscale('log')  # Log scale if the range is large
    plt.title('Performance Distribution by Configuration')
    plt.savefig(path / "plots" / "performance.pdf")
    plt.close()

    return stats

def evaluate_results(path, model: str = None):
    df = pd.read_csv(path)
    if model is not None:
        df = df[df["model"] == model]
    else:
        df_sorted = df[(df['configuration'] == 'Top') | (df['configuration'] == 'Default')].sort_values(by="p-value", ascending=True)
        print(df_sorted.to_string())

    number_total = len(df[df['configuration'] == 'Top'])
    number_sig = len(df[(df['configuration'] == 'Top') & (df['p-value'] < 0.05)])
    print(f"{number_sig}/{number_total} results are significant")

    number_improvment = 0
    number_improvement_sig = 0
    grouped = df.groupby(['model', 'dim_bud'])
    for name, group in grouped:
        default_mean = group[group['configuration'] == 'Default']['mean'].values[0]
        top_mean = group[group['configuration'] == 'Top']['mean'].values[0]
        if top_mean < default_mean:
            number_improvment += 1
            if group[group['configuration'] == 'Top']['p-value'].values[0] <= 0.05:
                number_improvement_sig += 1
    print(f"In {number_improvment} cases the found configuration achieved better results, {number_improvement_sig} time these results are significant")


def measure_time(paths: list[Path]):
    results = {}
    for path in paths:
        for root, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename == "optimization.json":
                    full_path = Path(os.path.join(root, filename))
                    model = full_path.parent.parent.parent.name
                    dim_bud = full_path.parent.parent.parent.parent.name
                    bud = dim_bud[2:5]
                    dim = dim_bud[9:]
                    # _, bud, _, _, dim = dim_bud.split("_")
                    if not model in results:
                        results[model] = {}
                    if not dim in results[model]:
                        results[model][dim] = {}
                    with open(full_path, 'r') as file:
                        data = json.load(file)
                        total_time = data["used_walltime"]
                        function_time = data["used_target_function_walltime"]
                    results[model][dim][bud] = {"time": total_time, "function_time": function_time}
    
    for model in results.keys():
        model_results = results[model]
        total_time = 0
        total_func_time = 0
        for dim, bud_dict in model_results.items():
            dim_time = 0
            dim_func_time = 0
            for bud, time_dict in bud_dict.items():
                dim_time += time_dict["time"]
                dim_func_time += time_dict["function_time"]
            results[model][dim]["time"] = dim_time
            results[model][dim]["function_time"] = dim_func_time
            total_time += dim_time
            total_func_time += dim_func_time
        results[model]["time"] = total_time
        results[model]["function_time"] = total_func_time

    output_dir = Path("Times/Multi_2351015")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "times.json"

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse results.')
    parser.add_argument('--analyse', type=bool, help='Name of the result folder', required=False, default=False)
    parser.add_argument('--eval', type=bool, help='Whether to evaluate the results', required=False, default=False)
    args = parser.parse_args()

    path = "Output/Multi_Dim_2351015"
    paths = ["Output/Final_D235_B235", "Output/Final_D235_B1015", "Output/Final_D1015_B235", "Output/Final_D1015_B1510"]

    # measure_time([path])

    if args.analyse:
        results = []
        for entry in os.listdir(path):
            if os.path.isdir(os.path.join(path, entry)) and entry.startswith("B_"):
                budget_dimension_folder = os.path.join(path, entry)
                for entry in os.listdir(budget_dimension_folder):
                    if os.path.isdir(os.path.join(budget_dimension_folder, entry)):
                        model_folder = os.path.join(budget_dimension_folder, entry)
                        print(f"{os.path.basename(budget_dimension_folder)}, {entry}")
                        df = analyse_results(Path(model_folder))
                        df["model"] = entry
                        df["dim_bud"] = os.path.basename(budget_dimension_folder)
                        results.append(df)
        final = pd.concat(results)
        final.to_csv(Path(path) / "significance.csv")

    if args.eval:
        path_to_sig = Path(path) / "significance.csv"
        # General results
        evaluate_results(path_to_sig)
        # per Modell
        models = ["MetaModel", "MetaModelOnePlusOne", "ChainMetaModelPowell", "CMA", "MetaModelFmin2"]
        for model in models:
            print()
            print("For model: ", model)
            evaluate_results(path_to_sig, model)

