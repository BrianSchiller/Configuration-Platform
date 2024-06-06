from smac import AlgorithmConfigurationFacade as ACFacade
import matplotlib.pyplot as plt

from settings import Settings

class Tester():
    def __init__(
        self, smac: ACFacade
    ) -> None:
        self.smac = smac

    def test(self, top_configs, test_configs, output, iterations = None):
        if iterations is None:
            iterations = Settings.test_iterations

        top_values = self.get_values(top_configs, iterations)
        test_values = self.get_values(test_configs, iterations)

        plt.clf()

        for key, value in top_values.items():
            plt.plot(value, color='red', label='Trained Configs' if key == 0 else '')

        for key, value in test_values.items():
            plt.plot(value, color='blue', label='Test Configs' if key == 0 else '')

        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (Loss)')
        plt.title('Optimization Process')
        output_file = output / f"test.png"
        plt.savefig(output_file)

        return 
    
    def get_values(self, configs, iterations):
        values = {}
        # validate each config -iterations- times and add mean to the values
        for index, config in enumerate(configs):
            values[index] = []
            for _ in range(iterations):
                performance = self.smac.validate(config)
                values[index].append(performance)
        return values
