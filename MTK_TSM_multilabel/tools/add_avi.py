# 設定文件路徑
file_path = "/data/ivs01/MTK_TSM/mtk_dms_data_20240712_label/test.txt"

# 讀取文件
with open(file_path, 'r') as file:
    lines = file.readlines()

# 處理每一行的內容
new_lines = []
for line in lines:
    parts = line.split()  # 默認按空格分割
    video_path = parts[0] + ".avi"  # 加上 .avi 擴展名
    new_lines.append(video_path + "\n")  # 僅保存影片路徑

# 覆蓋原始文件或保存到新文件
with open(file_path, 'w') as file:
    file.writelines(new_lines)