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
base_path = "/data/ivs01/MTK_TSM/mtk_dms_test/mtk_dms_data_test_new/"
new_path = "/data/ivs01/MTK_TSM/mtk_dms_test/mtk_dms_data_test_label/"
if not os.path.exists(new_path):
    os.makedirs(new_path)

# 讀取class label檔案
label_path = "/data/ivs01/MTK_TSM/mtk_dms_test/mtk_dms_data_test/classRef.txt"
categories = []
with open(label_path, 'r') as f:
    for line in f:
        idx, category = line.strip().split()
        categories.append((int(idx), category))

shutil.copy(label_path, new_path)
with open(label_path, 'r', encoding='utf-8') as fin, open(new_path + "classInd.txt", 'w', encoding='utf-8') as fout:
    # 逐行处理输入文件
    for line in fin:
        # 提取每一行的数字标签
        label = line.split()[0]
        # 写入数字标签到输出文件
        fout.write(label + '\n')

train_list = []
val_list = defaultdict(list)



for idx, category in categories:
    category_path = os.path.join(base_path, category)
    org_videos = [video for video in os.listdir(category_path) if os.path.join(category_path, video)] # 所有影片

    random.shuffle(org_videos)  # 對影片進行隨機排序

    num_all_videos = len(org_videos)
    num_val_needed = int(num_all_videos * 0.2)   ############ 如果要沿用舊的val.txt
    num_val_needed = max(num_val_needed, 0)  # 確保需求的驗證數不為負

    if idx >= 16:
        # 类别>=16的全部归为训练集(Phone_Drowsy 、 Drowsy_Phone)
        num_val_needed = 0
        num_train = num_all_videos
    else:
        num_train = num_all_videos - num_val_needed 

    print("class:", idx)
    print("num_all_videos", num_all_videos)
    print("num_train", num_train)
    print("num_val_needed", num_val_needed)

    train_videos = org_videos[:num_train]
    val_videos = org_videos[num_train:num_train + num_val_needed]  # 剩下的作為驗證集

    for video in train_videos:
        video_path = os.path.join(category_path, video)
        frame_count = get_frame_count(video_path)
        if idx >= 16:
            # 类别>=16的在train.txt中标签减去12
            line = f"{video_path} {frame_count} {idx - 12}\n"
        else:
            line = f"{video_path} {frame_count} {idx}\n"
        train_list.append(line)

    for video in val_videos:
        video_path = os.path.join(category_path, video)
        frame_count = get_frame_count(video_path)
        line = f"{video_path} {frame_count} {idx}\n"
        val_list[idx].append(line)

# 將訓練和驗證列表寫入到對應的文件中
train_file_path = os.path.join(new_path, "train.txt")
with open(train_file_path, "w") as train_file:
    for line in train_list:
        train_file.write(line)

# 先寫入現有的 val.txt 文件內容，再追加新的驗證內容
val_file_path = os.path.join(new_path, "val.txt")
with open(val_file_path, "w") as val_file:
    for idx, lines in val_list.items():
        for line in lines:
            val_file.write(line)

test_file_path = os.path.join(new_path, "test.txt")
with open(test_file_path, "w") as test_file:
    for idx, lines in val_list.items():
        for line in lines:
            test_file.write(line)

