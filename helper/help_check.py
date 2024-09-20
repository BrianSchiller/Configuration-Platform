import os

def check_for_plots_folder(main_folder):
    missing_plots_paths = []

    # Walk through the main folder
    for entry in os.listdir(main_folder):
        if os.path.isdir(os.path.join(main_folder, entry)):
            budget_dimension_folder = os.path.join(main_folder, entry)
            for entry in os.listdir(budget_dimension_folder):
                if os.path.isdir(os.path.join(budget_dimension_folder, entry)):
                    model_folder = os.path.join(budget_dimension_folder, entry)
                    if os.path.isdir(model_folder):
                        # Check for the 'plots' subdirectory
                        plots_folder = os.path.join(model_folder, 'plots')
                        if not os.path.exists(plots_folder):
                            missing_plots_paths.append(model_folder)
    
    return missing_plots_paths

def main():
    main_folder = 'Output/Final_D1015_B1510'  # Replace with the path to your main folder
    missing_plots = check_for_plots_folder(main_folder)
    
    if missing_plots:
        print("The following paths are missing a 'plots' folder:")
        for path in missing_plots:
            print(path)
    else:
        print("All folders contain a 'plots' folder.")

if __name__ == "__main__":
    main()
