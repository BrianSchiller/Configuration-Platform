import os
from pathlib import Path

root_dir = "Output/Multi_Dim_2351015"

x_values = [200, 300, 500]
y_values = [[2,3,5,10, 15]]

configs = {}
defaults = Path("default")
txt_files = list(defaults.glob("*.txt"))

for file in txt_files:
    model = file.name.replace(".txt", "")
    with file.open("r") as file:
        config = file.read()
    configs[model] = {
        "default": {
            "default": config
        }
    }

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
            if not y_str in configs[model_dir.name]:
                configs[model_dir.name][y_str] = {}
            
            # Read txt files
            txt_files = list(model_dir.glob("*.txt"))
            if len(txt_files) > 0:
                with txt_files[0].open("r") as file:
                    config = file.read()
                    configs[model_dir.name][y_str][x] = config

output_dir = Path(os.path.join(root_dir, "configs"))
output_dir.mkdir(parents=True, exist_ok=True)

for model, y_dict in configs.items():
    model_file_path = output_dir / f"{model}.txt"
    with model_file_path.open("w") as model_file:
        for y_str, x_dict in y_dict.items():
            for x, config in x_dict.items():
                model_file.write(f"Budget: {x}, Dimension: {y_str}\n")
                model_file.write(config + "\n")

