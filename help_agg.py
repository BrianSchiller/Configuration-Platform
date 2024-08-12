import os
import shutil
from pathlib import Path

def copy_tree(src, dest, exclude_exts=None):
    """
    Copy the contents of src to dest, excluding files with extensions in exclude_exts.
    
    :param src: Source directory.
    :param dest: Destination directory.
    :param exclude_exts: Set of file extensions to exclude from copying.
    """
    if exclude_exts is None:
        exclude_exts = set()

    # Ensure destination directory exists
    if not os.path.exists(dest):
        os.makedirs(dest)

    # Traverse source directory
    for root, dirs, files in os.walk(src):
        # Compute relative path from source
        rel_path = os.path.relpath(root, src)
        dest_path = os.path.join(dest, rel_path)

        # Ensure destination subdirectories exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        # Copy files
        for file in files:
            if not any(file.endswith(ext) for ext in exclude_exts):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, file)
                shutil.copy2(src_file, dest_file)

def main(source_dirs, destination_dir):
    """
    Copy all directories and files from multiple source directories to a single destination,
    excluding .json files.
    
    :param source_dirs: List of source directories.
    :param destination_dir: Destination directory.
    """
    exclude_exts = {'.json'}
    
    # Iterate through all source directories and copy contents
    for src_dir in source_dirs:
        print(f"Moving {src_dir}")
        if os.path.isdir(src_dir):
            copy_tree(src_dir, destination_dir, exclude_exts)
        else:
            print(f"Source directory {src_dir} does not exist or is not a directory.")

if __name__ == "__main__":
    source_directories = [
        "Output/Final_D235_B235",
        "Output/Final_D235_B1015",
        "Output/Final_D1015_B235",
        "Output/Final_D1015_B1510"
    ]
    destination_directory = "Output/Final"

    main(source_directories, destination_directory)
