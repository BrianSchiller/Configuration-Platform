from smac import AlgorithmConfigurationFacade as ACFacade
import matplotlib.pyplot as plt
import statistics

from settings import Settings

class Tester():
    def __init__(
        self, smac: ACFacade
    ) -> None:
        self.smac = smac

    def test(self, top_configs, test_configs, default_config, output, iterations = None):
        if iterations is None:
            self.iterations = Settings.test_iterations

        top_values = self.get_values(top_configs)
        test_values = self.get_values(test_configs)
        default_values = self.get_values([default_config])

        plt.clf()

        for key, value in top_values.items():
            plt.plot(value, color='red', label='Trained Configs' if key == 0 else '')

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

        min_mean = float('inf')
        key_with_min_mean = None

        # Return the config with the lowest overall performance
        for key, value_list in top_values.items():
            current_mean = statistics.mean(value_list)
            if current_mean < min_mean:
                min_mean = current_mean
                key_with_min_mean = key

        return top_configs[key_with_min_mean]
    
    def get_values(self, configs):
        values = {}
        # validate each config -iterations- times and add mean to the values
        for index, config in enumerate(configs):
            values[index] = []
            for _ in range(self.iterations):
                performance = self.smac.validate(config)
                values[index].append(performance)
        return values
