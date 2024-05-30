import os
import py7zr
import math

def split_7z_file(input_7z_file, output_dir='split_files', max_size_mb=24):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the 7z file
    with py7zr.SevenZipFile(input_7z_file, mode='r') as z:
        all_files = z.getnames()
        z.extractall(path=output_dir)

    # Calculate the number of parts needed
    file_sizes = {file: os.path.getsize(os.path.join(output_dir, file)) for file in all_files}
    total_size = sum(file_sizes.values())
    num_splits = math.ceil(total_size / (max_size_mb * 1024 * 1024))

    # Sort files by size (largest first)
    sorted_files = sorted(file_sizes.items(), key=lambda item: item[1], reverse=True)

    # Split files into parts
    current_part = 1
    current_size = 0
    current_files = []

    for file, size in sorted_files:
        if current_size + size > max_size_mb * 1024 * 1024:
            # Create a new 7z file for the current part
            part_name = os.path.join(output_dir, f'part_{current_part}.7z')
            with py7zr.SevenZipFile(part_name, 'w') as archive:
                for f in current_files:
                    archive.write(os.path.join(output_dir, f), f)
            # Prepare for the next part
            current_part += 1
            current_size = 0
            current_files = []

        current_files.append(file)
        current_size += size

    # Create the last part
    if current_files:
        part_name = os.path.join(output_dir, f'part_{current_part}.7z')
        with py7zr.SevenZipFile(part_name, 'w') as archive:
            for f in current_files:
                archive.write(os.path.join(output_dir, f), f)

    # Clean up the extracted files
    for file in all_files:
        os.remove(os.path.join(output_dir, file))

# Example usage
split_7z_file('your_large_file.7z')


#########


import os
import py7zr
import shutil

def merge_7z_files(split_dir='split_files', output_file='merged_file.7z', temp_dir='temp_merge'):
    # Ensure the temporary directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract each split file into the temporary directory
    split_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.7z')])
    for split_file in split_files:
        split_file_path = os.path.join(split_dir, split_file)
        with py7zr.SevenZipFile(split_file_path, mode='r') as z:
            z.extractall(path=temp_dir)
    
    # Create a new 7z archive with the combined contents
    with py7zr.SevenZipFile(output_file, 'w') as archive:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                archive.write(file_path, os.path.relpath(file_path, temp_dir))
    
    # Clean up the temporary files
    shutil.rmtree(temp_dir)

# Example usage
merge_7z_files()


import os

def split_file(input_file, output_dir='split_files', chunk_size_mb=24):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    
    with open(input_file, 'rb') as f:
        chunk_number = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_file = os.path.join(output_dir, f'{os.path.basename(input_file)}.part{chunk_number:04d}')
            with open(chunk_file, 'wb') as chunk_f:
                chunk_f.write(chunk)
            chunk_number += 1

# Example usage
split_file('model.safetensors')




import os

def merge_files(input_dir='split_files', output_file='merged_model.safetensors'):
    # Get a list of all chunk files
    chunk_files = sorted([f for f in os.listdir(input_dir) if f.startswith('model.safetensors.part')])
    
    if not chunk_files:
        print("No chunk files found. Please check the input directory.")
        return

    print("Found chunk files:", chunk_files)
    
    with open(output_file, 'wb') as merged_f:
        for chunk_file in chunk_files:
            print(f"Merging {chunk_file}...")
            chunk_file_path = os.path.join(input_dir, chunk_file)
            with open(chunk_file_path, 'rb') as chunk_f:
                merged_f.write(chunk_f.read())
    
    print(f"Successfully merged files into {output_file}")

# Example usage
merge_files()
