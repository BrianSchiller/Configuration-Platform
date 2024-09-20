import pandas as pd
from scipy.stats import pearsonr, pointbiserialr
from pathlib import Path
import os
import ast
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def read_configs():
    root_dir = "Output/Multi_Dim_2351015"

    x_values = [200, 300, 500]
    y_values = [[2, 3, 5, 10, 15]]

    configs = {}

    for x in x_values:
        for y in y_values:
            if isinstance(y, list):
                y_str = '_'.join(map(str, y))
            else:
                y_str = str(y)
            bud_dim_dir = Path(os.path.join(root_dir, f'B_{x}__D_{y_str}'))
            
            # Iterate over model folders
            model_dirs = [f for f in bud_dim_dir.iterdir() if f.is_dir()]
            for model_dir in model_dirs:
                # Make sure structure exists
                if not model_dir.name in configs:
                    configs[model_dir.name] = {}
                if not y_str in configs[model_dir.name]:
                    configs[model_dir.name][y_str] = {}
                
                # Read txt files
                txt_files = list(model_dir.glob("*.txt"))
                if len(txt_files) > 0:
                    with txt_files[0].open("r") as file:
                        config = file.read()
                        configs[model_dir.name][y_str][x] = config

    return configs

def get_df(dict, var):
    dimension_list = []
    budget_list = []
    var_list = []

    for dimension, budgets in dict.items():
        for budget, details in budgets.items():
            # Convert the string representation of the dictionary to a dictionary
            details_dict = ast.literal_eval(details)
            # Append data to lists
            dimension_list.append(dimension)
            budget_list.append(budget)
            var_list.append(details_dict.get(var))

    # Create DataFrame
    df = pd.DataFrame({
        # 'Dimension': dimension_list,
        'Budget': budget_list,
        f'{var}': var_list
    })

    return df

def get_correlation(configs):
    for model, dimensions in configs.items():
        print()
        print(model)
        print()
        for dimension, budgets in dimensions.items():
            for budget, details in budgets.items():
                details_dict = ast.literal_eval(details)
                for key, value in details_dict.items():
                    if isinstance(value, bool):
                        df = get_df(dimensions, key)
                        calculate_corr(df, key, "bool")
                    elif isinstance(value, (int, float)):
                        df = get_df(dimensions, key)
                        calculate_corr(df, key, "numerical")
                    elif isinstance(value, str) and key != "noise_handling":
                        df = get_df(dimensions, key)
                        calculate_corr(df, key, "categorical")
                break
            break
        print()

def calculate_corr(df, val, type):
    # df['Dimension'] = pd.to_numeric(df['Dimension'], errors='coerce')
    df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')

    if type == "numerical":
        df[val] = pd.to_numeric(df[val], errors='coerce')

        corr_budget, _ = pearsonr(df[val], df['Budget'])
        # corr_dimension, _ = pearsonr(df[val], df['Dimension'])

        print(f"Pearson correlation coefficient between {val} and budget: {corr_budget:.4f}")
        # print(f"Pearson correlation coefficient between {val} and dimension: {corr_dimension:.4f}")

    if type == "bool":
        df[val] = df[val].astype(int)
        corr_budget, _ = pointbiserialr(df[val], df['Budget'])
        # corr_dimension, _ = pointbiserialr(df[val], df['Dimension'])

        print(f"Point Biseral correlation coefficient between {val} and budget: {corr_budget:.4f}")
        # print(f"Point Biseral coefficient between {val} and dimension: {corr_dimension:.4f}")

    if type == "categorical":
        df[val] = df[val].astype('category')
        calculate_eta_squared(df, val, "Budget")
        # calculate_eta_squared(df, val, "Dimension")
        
def calculate_eta_squared(df, categorical_var, numerical_var):
    # Fit the ANOVA model
    if df[categorical_var].nunique() <= 1:
        print(f"Not enough variation in the independent variable '{categorical_var}' to perform ANOVA.")
        return
    formula = f"{numerical_var} ~ C({categorical_var})"
    model = ols(formula, data=df).fit()
    anova_results = anova_lm(model, typ=2)
    
    # Extract sum of squares
    ss_between = anova_results['sum_sq']['C(' + categorical_var + ')']
    ss_within = anova_results['sum_sq']['Residual']
    
    # Calculate Eta-Squared
    eta_squared = ss_between / (ss_between + ss_within)
    
    print(f"Eta-Squared (η²) between {categorical_var} and {numerical_var}: {eta_squared:.4f}")

if __name__ == "__main__":
    configs = read_configs()
    get_correlation(configs)
