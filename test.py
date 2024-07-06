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
        result_output = output / "test_results"

        self.top_results = result_output / "top_results"
        top_values = self.get_values(top_configs, self.top_results)
        print("Finished Testing of found configurations")

        self.sample_results = result_output / "sample_results"
        test_values = self.get_values(test_configs, self.sample_results)
        print("Finished Testing of sampled configurations")

        self.default_results = result_output / "default_results"
        default_values = self.get_values([default_config], self.default_results)
        print("Finished Testing of default configuration")

        output = output / "plots"
        self.plot_performance_history(top_values, test_values, default_values, output)
        self.plot_problem_performance(output)

        return
    
    def get_values(self, configs, out_dir):
        values = {}
        # validate each config -iterations- times
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
            row_index = int(prob) - 1
            
            # Group by source and config within the current (dimension, budget, problem, instance) group
            temp_data = grouped.get_group((dim, bud, prob, inst))
            Q1 = temp_data['loss'].quantile(0.25)
            Q3 = temp_data['loss'].quantile(0.75)
            IQR = Q3 - Q1
            threshold = Q3 + 1.5 * IQR  # You can adjust the multiplier if needed
            temp_data.loc[:, 'loss'] = temp_data['loss'].clip(upper=threshold)
            group_data = temp_data.groupby(['source', 'config'])
            counts = group_data.size()

            # Plot each source x config combination
            for (source, config), values in group_data:
                if source == "top":
                    # Use different shades of red for each "top" config
                    color = (1 - int(config) / counts.loc[(source, config)], 0, 0)
                    axs[row_index * num_cols + (inst - 1)].plot(range(counts.loc[(source, config)]), values['loss'], color=color, label=f"Best, Config: {config}")
                elif source == "default":
                    axs[row_index * num_cols + (inst - 1)].plot(range(counts.loc[(source, config)]), values['loss'], color='green', label='Default Config')
                elif source == "sample" and config == "0":
                    axs[row_index * num_cols + (inst - 1)].plot(range(counts.loc[(source, config)]), values['loss'], color='blue', label='Sample Configs')
                else:
                    axs[row_index * num_cols + (inst - 1)].plot(range(counts.loc[(source, config)]), values['loss'], color='blue')

            axs[row_index * num_cols + (inst - 1)].axhline(y=threshold, color='gray', linestyle='--', linewidth=1)
            axs[row_index * num_cols + (inst - 1)].text(0, threshold, f'Threshold: {threshold:.2f}', color='gray', verticalalignment='bottom')
            
            # Set labels and title for each subplot
            axs[row_index * num_cols + (inst - 1)].set_xlabel('Index')
            axs[row_index * num_cols + (inst - 1)].set_ylabel('Loss')
            axs[row_index * num_cols + (inst - 1)].set_title(f'Problem: {prob}, Instance: {inst}')
            axs[row_index * num_cols + (inst - 1)].legend()
        
        # Adjust layout and spacing
        plt.tight_layout()
        plt.savefig(outdir / 'problems.pdf')

        
    def get_problem_results(self, directory):
        data = []
        file_paths = []
        # Walk through each directory and subdirectory to collect file paths
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    file_paths.append((root, file))
        # Sort the file paths based on the filenames
        file_paths.sort(key=lambda x: x[1])
        for root, file in file_paths:
            filepath = os.path.join(root, file)
            df = pd.read_csv(filepath)
            # Extract 'source' and 'config' from directory and file names
            source = os.path.basename(root).split('_')[0]
            config = file.split('_')[0]
            df['source'] = source
            df['config'] = config
            # Append the DataFrame to the list
            data.append(df)
        return pd.concat(data, ignore_index=True)
    
    
    def plot_performance_history(self, top_values, test_values, default_values, output):
        plt.clf()

        for key, value in top_values.items():
            if key == 0:
                plt.plot(value, color='salmon', label='1. Trained Config')
            if key == 1:
                plt.plot(value, color='red', label='2. Trained Config')
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
        os.makedirs(output, exist_ok=True)
        output_file = output / f"test.pdf"
        plt.savefig(output_file)

