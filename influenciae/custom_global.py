from base_analyzer import AbstractGlobalInfluence

class CustomGlobalAnalyzer(AbstractGlobalInfluence):
    def setup_calculator(self, model_path, train_dataset):
        # 1. Load model or checkpoints
        # 2. Configure loss and wrap the custom Deel influence calculator, or make sure it has a .top_k method with compatible structure
        # 3. Return the calculator
        return MyCustomCalculator(...)

if __name__ == "__main__":
    analyzer = CustomGlobalAnalyzer(k_display=10)
    analyzer.run()