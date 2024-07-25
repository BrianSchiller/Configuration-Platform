import os
import shutil

# Define the source and destination directories
old_root_dir = 'Output/_Final-D1015-B235'
new_root_dir = '../AS4BBO/configurations'

# List of x and y values based on your description
x_values = [200, 300, 500]
y_values = [10, 15]

# Ensure the new directory structure exists
for x in x_values:
    for y in y_values:
        new_dir_path = os.path.join(new_root_dir, str(x), str(y))
        os.makedirs(new_dir_path, exist_ok=True)

# Iterate over the old directory structure
for x in x_values:
    for y in y_values:
        old_dir_path = os.path.join(old_root_dir, f'B_{x}__D_{y}')
        new_dir_path = os.path.join(new_root_dir, str(x), str(y))

        print(old_dir_path)

        if os.path.exists(old_dir_path):
            for subdir, _, files in os.walk(old_dir_path):
                for file in files:
                    if file.endswith('.txt'):
                        old_file_path = os.path.join(subdir, file)
                        new_file_path = os.path.join(new_dir_path, file)
                        shutil.copy(old_file_path, new_file_path)

print("Files have been moved successfully!")
