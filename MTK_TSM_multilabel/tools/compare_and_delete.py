import os

# 定義資料夾 A 和 B 的路徑
folder_A = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Eating_2sec/"  # 資料夾 A
folder_B = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Training/Eating/"  # 資料夾 B

# 取得資料夾 B 中的所有影片檔名
files_in_B = set(os.listdir(folder_B))

# 遍歷資料夾 A，刪除在 B 中出現的檔案
for file in os.listdir(folder_A):
    # 檢查檔案是否是影片檔案
    if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
        # 如果影片在資料夾 B 中存在，就刪除資料夾 A 中的對應影片
        if file in files_in_B:
            file_path = os.path.join(folder_A, file)
            try:
                os.remove(file_path)
                print(f"已刪除 {file}")
            except Exception as e:
                print(f"無法刪除 {file}: {e}")
