import os
import shutil

# --- Configuration ---
# Replace with the path to the directory containing 'diggers', 'gen', and 'wgen'
base_path = "/home/mohammad/Documents/uni/deeplearning/FinalProject/data/sr_outputs/content/sr_outputs" 
# The name of the new folder that will contain the combined structure
target_folder_name = "combined_frames" 

source_folders = ["diggers_official", "genivsr", "wgen68ivsr"]
frame_name = "00000005.png"
num_subfolders = 30 # Folders "000" to "029"

# --- Script Logic ---

# 1. Define the full path for the new target folder
target_path = os.path.join(base_path, target_folder_name)

# 2. Create the target folder and its 30 subfolders if they don't exist
os.makedirs(target_path, exist_ok=True)
for i in range(num_subfolders):
    sub_folder_name = f"{i:03d}"  # Formats numbers 0 to 29 as "000" to "029"
    os.makedirs(os.path.join(target_path, sub_folder_name), exist_ok=True)

# 3. Iterate through the source folders and copy/rename files
for source_folder in source_folders:
    source_prefix = source_folder[:3] # 'dig', 'gen', 'wge' (we'll fix this in the rename step)
    
    # Correctly set the desired prefix for the new file name
    if source_folder == "diggers_official":
        prefix = "dig"
    elif source_folder == "genivsr":
        prefix = "gen"
    elif source_folder == "wgen68ivsr":
        prefix = "wgen"
    else:
        continue # Skip if an unexpected folder name is encountered

    full_source_folder_path = os.path.join(base_path, source_folder)

    for i in range(num_subfolders):
        sub_folder_name = f"{i:03d}"
        
        # Define source and destination paths
        original_frame_path = os.path.join(full_source_folder_path, sub_folder_name, frame_name)
        
        # New file name: e.g., "dig_00000005.png"
        new_file_name = f"{prefix}_{frame_name}"
        destination_frame_path = os.path.join(target_path, sub_folder_name, new_file_name)

        # Copy and rename the file
        if os.path.exists(original_frame_path):
            shutil.copy2(original_frame_path, destination_frame_path)
        else:
            print(f"Warning: File not found at {original_frame_path}")

print(f"Content successfully combined and moved to: {target_path}")