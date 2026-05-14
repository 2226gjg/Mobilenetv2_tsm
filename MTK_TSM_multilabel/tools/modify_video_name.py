import os

def rename_files(directory):
    for root, dirs, files in os.walk(directory):
        for name in files:
            if " " in name:
                # 构造原始文件的完整路径
                old_file_path = os.path.join(root, name)
                # 构造新的文件名，替换空格为下划线
                new_name = name.replace(" ", "_")
                # 构造新文件的完整路径
                new_file_path = os.path.join(root, new_name)
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{old_file_path}' to '{new_file_path}'")

# 替换为你的目标目录
# target_directory = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Training/"
target_directory = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Drama/drama_video/"
rename_files(target_directory)
