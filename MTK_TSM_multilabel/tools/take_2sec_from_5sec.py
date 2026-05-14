import os
import shutil

# 定義來源資料夾與目標資料夾
source_folder = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Eating_5sec_1/"  # 主資料夾
destination_folder = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Eating_2sec/"  # 目標資料夾

# 創建目標資料夾（如果不存在）
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍歷主資料夾內的子資料夾
for subdir, dirs, files in os.walk(source_folder):
    for file in files:
        # 檢查檔案是否為影片（根據副檔名來判斷）
        if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            source_file = os.path.join(subdir, file)
            destination_file = os.path.join(destination_folder, file)
            try:
                # 移動影片檔案到目標資料夾
                shutil.copy(source_file, destination_file)
                print(f"已copy {file} 到 {destination_folder}")
            except Exception as e:
                print(f"無法copy {file}: {e}")
