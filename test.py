from smac import AlgorithmConfigurationFacade as ACFacade
import matplotlib.pyplot as plt
import statistics
import os
import pandas as pd

import settings

class Tester():
    def __init__(
        self, smac: ACFacade
    ) -> None:
        self.smac = smac

    def test(self, top_configs, test_configs, default_config, output, iterations = None):
        if iterations is None:
            self.iterations = settings.test_iterations
        
        settings.store_problem_results = True
        self.top_results = output / "top_results"
        top_values = self.get_values(top_configs, self.top_results)
        self.sample_results = output / "sample_results"
        test_values = self.get_values(test_configs, self.sample_results)
        self.default_results = output / "default_results"
        default_values = self.get_values([default_config], self.default_results)

        self.plot_performance_history(top_values, test_values, default_values, output)
        self.plot_problem_performance(output)

        min_mean = float('inf')
        key_with_min_mean = None

        # Return the config with the lowest overall performance
        for key, value_list in top_values.items():
            current_mean = statistics.mean(value_list)
            if current_mean < min_mean:
                min_mean = current_mean
                key_with_min_mean = key

        return top_configs[key_with_min_mean]
    
    def get_values(self, configs, out_dir):
        values = {}
        # validate each config -iterations- times and add mean to the values
        for index, config in enumerate(configs):
            values[index] = []
            for _index in range(self.iterations):
                settings.problem_result_dir = out_dir / f"{index}_{_index}.csv"
                performance = self.smac.validate(config)
                values[index].append(performance)
        return values
    
    
    def plot_problem_performance(self, outdir):
        directories = [self.top_results, self.default_results, self.sample_results]
        data = pd.concat([self.get_problem_results(dir) for dir in directories], ignore_index=True)

        grouped = data.groupby(['dimension', 'budget', 'problem', 'instance'])
    
        num_cols = len(data['instance'].unique())  
        num_rows = len(data['problem'].unique())  

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(7*num_cols, 7*num_rows))
        fig.subplots_adjust(hspace=1.5)

        if num_rows > 1:
            axs = axs.flatten()

        # Iterate over each (dimension, budget, problem, instance) group
        for i, (dim, bud, prob, inst) in enumerate(grouped.groups.keys()):
            row_index = data['problem'].unique().tolist().index(prob)
            
            # Group by source and config within the current (dimension, budget, problem, instance) group
            group_data = grouped.get_group((dim, bud, prob, inst))
            group_data_grouped = group_data.groupby(['source', 'config'])
            counts = group_data_grouped.size()

            # Plot each source x config combination
            for (source, config), source_group_data in group_data_grouped:
                if source == "top":
                    # Use different shades of red for each "top" config
                    color = (1 - int(config) / counts.loc[(source, config)], 0, 0)
                    axs[row_index * num_cols + inst].plot(range(counts.loc[(source, config)]), source_group_data['loss'], color=color, label=f"Best, Config: {config}")
                elif source == "default":
                    axs[row_index * num_cols + inst].plot(range(counts.loc[(source, config)]), source_group_data['loss'], color='green', label='Default Config')
                elif source == "sample" and config == "0":
                    axs[row_index * num_cols + inst].plot(range(counts.loc[(source, config)]), source_group_data['loss'], color='blue', label='Sample Configs')
                else:
                    axs[row_index * num_cols + inst].plot(range(counts.loc[(source, config)]), source_group_data['loss'], color='blue')
            
            # Set labels and title for each subplot
            axs[row_index * num_cols + inst].set_xlabel('Index')
            axs[row_index * num_cols + inst].set_ylabel('Loss')
            axs[row_index * num_cols + inst].set_title(f'Problem: {prob}, Instance: {inst}')
            axs[row_index * num_cols + inst].legend()
        
        # Adjust layout and spacing
        plt.tight_layout()
        plt.savefig(outdir / 'problems.pdf')  # Adjust the file extension and path as needed

        
    def get_problem_results(self, directory):
        data = []
        for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".csv"):
                        filepath = os.path.join(root, file)
                        df = pd.read_csv(filepath)
                        df['source'] = os.path.basename(root).split('_')[0]
                        df['config'] = file.split('_')[0]
                        data.append(df)
        return pd.concat(data, ignore_index=True)
    
    
    def plot_performance_history(self, top_values, test_values, default_values, output):
        plt.clf()

        for key, value in top_values.items():
            if key == 0:
                plt.plot(value, color='salmon', label='1. Trained Configs')
            if key == 1:
                plt.plot(value, color='red', label='2. Trained Configs')
            if key > 1:
                plt.plot(value, color='darkred', label='Trained Configs' if key == 2 else "")

        for key, value in test_values.items():
            plt.plot(value, color='blue', label='Sample Configs' if key == 0 else '')

        for key, value in default_values.items():
            plt.plot(value, color='green', label='Default Config' if key == 0 else '')

        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (Loss)')
        plt.title('Optimization Process')
        output_file = output / f"test.png"
        plt.savefig(output_file)

