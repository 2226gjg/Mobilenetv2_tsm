import os
import cv2
import random
import shutil
from collections import defaultdict

def get_frame_count(video_path):
    """返回影片的 frame 數"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

# 設定基礎資料夾路徑
base_path = "/data/ivs01/MTK_TSM/mtk_dms_20240924/mtk_dms_data_20240924_new/"
new_path = "/data/ivs01/MTK_TSM/mtk_dms_20240924/mtk_dms_data_20240924_label/"
if not os.path.exists(new_path):
    os.makedirs(new_path)
val_path = "/data/ivs01/MTK_TSM/mtk_dms_20240918/mtk_dms_data_20240918_label/val.txt"  # 上一版的val.txt

drama_path = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Drama/drama_video_new/"

# 讀取class label檔案
label_path = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/classRef.txt"
categories = []
with open(label_path, 'r') as f:
    for line in f:
        idx, category = line.strip().split()
        categories.append((int(idx), category))

with open(label_path, 'r', encoding='utf-8') as fin, open(new_path + "classRef.txt", 'w', encoding='utf-8') as fout:
    # 逐行处理输入文件
    for line in fin:
        # 提取每一行的数字标签
        label = line.split()[0]
        
        int_label = int(label)

        if int_label <= 15:
            fout.write(line)
        

with open(label_path, 'r', encoding='utf-8') as fin, open(new_path + "classInd.txt", 'w', encoding='utf-8') as fout:
    # 逐行处理输入文件
    for line in fin:
        # 提取每一行的数字标签
        label = line.split()[0]
        
        int_label = int(label)

        if int_label <= 14:
            # 写入数字标签到输出文件
            fout.write(label + '\n')
        if int_label == 15:
            # 写入数字标签到输出文件
            fout.write(label)

train_list = []
val_list = defaultdict(list)

# 讀取現有的 val.txt 文件內容
existing_val_set = defaultdict(list)
val_file_path = os.path.join(val_path)
if os.path.exists(val_file_path):
    with open(val_file_path, 'r') as val_file:
        for line in val_file:
            parts = line.strip().split()
            video_path = parts[0]
            frame_count = parts[1]
            idx = int(parts[2])
            if 'Drama' not in video_path:
                existing_val_set[idx].append((video_path, frame_count))


print(existing_val_set)

for idx, category in categories:
    category_path = os.path.join(base_path, category)
    org_videos = [video for video in os.listdir(category_path) if os.path.join(category_path, video)] # 所有影片
    videos = [video for video in os.listdir(category_path) if os.path.join(category_path, video) not in existing_val_set[idx]]  # 排除已在 val.txt 中的影片

    if idx < 16:
        category_drama_path = os.path.join(drama_path, category)
        drama_videos = [video for video in os.listdir(category_drama_path)]

    random.shuffle(videos)  # 對影片進行隨機排序

    num_existing_val = len(existing_val_set[idx])
    num_all_videos = len(org_videos)
    num_videos = len(videos)
    num_val_needed = int(num_all_videos * 0.2) - num_existing_val   ############ 如果要沿用舊的val.txt
    num_val_needed = max(num_val_needed, 0)  # 確保需求的驗證數不為負

    if idx >= 16:
        # 类别>=16的全部归为训练集(single_hand_multi 、 no_hand_multi)
        num_val_needed = 0
        # num_train = num_videos     #全放訓練
        num_train = 0                #全部不放訓練
    else:
        num_train = num_videos - num_val_needed - num_existing_val

    print("class:", idx)
    print("num_all_videos", num_all_videos)
    print("num_train", num_train)
    print("num_val", num_existing_val + num_val_needed)
    print("num_existing_val", num_existing_val)
    print("num_val_needed", num_val_needed)
    

    train_videos = videos[:num_train]
    val_videos = videos[num_train:num_train + num_val_needed]  # 剩下的作為驗證集

    for video in train_videos:
        video_path = os.path.join(category_path, video)
        frame_count = get_frame_count(video_path)
        if idx >= 16:
            # 类别>=16的在train.txt中标签减去12
            if(idx == 16 ):
                line = f"{video_path} {frame_count} {7}\n"
            elif(idx == 17):
                line = f"{video_path} {frame_count} {0}\n"
        else:
            line = f"{video_path} {frame_count} {idx}\n"
        train_list.append(line)

    for video in val_videos:
        video_path = os.path.join(category_path, video)
        frame_count = get_frame_count(video_path)
        line = f"{video_path} {frame_count} {idx}\n"
        val_list[idx].append(line)

    if idx < 16 :
        for video in drama_videos:    #劇本
            video_path = os.path.join(category_drama_path, video)
            frame_count = get_frame_count(video_path)
            line = f"{video_path} {frame_count} {idx}\n"
            val_list[idx].append(line)

# 將訓練和驗證列表寫入到對應的文件中
train_file_path = os.path.join(new_path, "train.txt")
with open(train_file_path, "w") as train_file:
    for line in train_list:
        modified_line = line.replace("_new/", "_frame/")
        modified_line = modified_line.replace(".avi", "")
        train_file.write(modified_line)


# 先寫入現有的 val.txt 文件內容，再追加新的驗證內容
val_file_path = os.path.join(new_path, "val.txt")
with open(val_file_path, "w") as val_file:
    for idx, entries in existing_val_set.items():
        for entry in entries:
            video_path, frame_count = entry
            val_file.write(f"{video_path} {frame_count} {idx}\n")
    for idx, lines in val_list.items():
        for line in lines:
            modified_line = line.replace("_new/", "_frame/")
            modified_line = modified_line.replace(".avi", "")
            val_file.write(modified_line)

test_file_path = os.path.join(new_path, "test.txt")
# 讀取 val.txt 的內容
with open(val_file_path, "r") as val_file:
    val_lines = val_file.readlines()

# 將處理後的內容寫入 test.txt
with open(test_file_path, "w") as test_file:
    for line in val_lines:
        # 分割每一行，僅保留第一個部分（video 路徑）
        video_path = line.split()[0]
        
        # 將 _frame/ 替換為 _new/，並在路徑末尾加上 .avi
        modified_line = video_path.replace("_frame/", "_new/") + ".avi"
        
        # 將處理後的內容寫入 test.txt
        test_file.write(modified_line + "\n")

