from smac import AlgorithmConfigurationFacade as ACFacade
import settings

class Validator():
    def __init__(
        self, smac: ACFacade
    ) -> None:
        self.smac = smac

    def validate(self, top_n: int = None, iterations: int = None):
        if top_n is None:
            top_n = settings.val_size
        if iterations is None:
            iterations = settings.val_iterations

        run_history = self.smac.runhistory

        configs = run_history.get_configs("cost")[:top_n]

        values = {}

        # validate each config -iterations- times and add mean to the values
        for index, config in enumerate(configs):
            total_value = 0
            for _ in range(iterations):
                # TODO Check whether validate returns different values
                # TODO Collect values for representation
                total_value += self.smac.validate(config)
            values[index] = total_value / iterations

        # Sort configurations based on the performance values (ascending order)
        sorted_indices = sorted(values, key=values.get)
        sorted_configs = [configs[index] for index in sorted_indices]

        return sorted_configs
