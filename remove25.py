import os
import shutil

if __name__ == '__main__':
    main_folders = ["/home/mohammad/Documents/uni/deeplearning/FinalProject/train_sharp2/train/train_sharp",
                "/home/mohammad/Documents/uni/deeplearning/FinalProject/train_sharp_bicubic2/train/train_sharp_bicubic/X4"]  # Replace with the path to your main folder

    for main_folder in main_folders:
        # print(main_folder)
        for folder_name in os.listdir(main_folder):
            # print(folder_name)
            folder_path = os.path.join(main_folder, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    
                        if file_name > "00000060":
                            file_path = os.path.join(folder_path, file_name)
                            try:
                                os.remove(file_path)
                            except OSError as e:
                                print(f"Error deleting {file_path}: {e}")