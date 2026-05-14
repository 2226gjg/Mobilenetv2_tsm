import os

def rename_directories(directory):
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            if " " in name:
                # 構造原始目錄的完整路徑
                old_dir_path = os.path.join(root, name)
                # 構造新的目錄名稱，替換空格為下劃線
                new_name = name.replace(" ", "_")
                # 構造新目錄的完整路徑
                new_dir_path = os.path.join(root, new_name)
                # 重命名目錄
                os.rename(old_dir_path, new_dir_path)
                print(f"Renamed directory '{old_dir_path}' to '{new_dir_path}'")

# 替換為你的目標目錄
target_directory = "/home/3105825/nova_action/mtk_dms_data_1219_frame/Seatbelt"
rename_directories(target_directory)