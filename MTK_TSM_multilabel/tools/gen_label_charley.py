import os
import cv2
import random
import shutil

def get_frame_count(video_path):
    """返回影片的 frame 數"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

# 設定基礎資料夾路徑
base_path = "/home/3105825/nova_action/mtk_dms_data_20240409_new/"
new_path = "/home/3105825/nova_action/mtk_dms_data_20240409_label/"
if not os.path.exists(new_path):
    os.makedirs(new_path)
categories = os.listdir(base_path)  # 列出所有的類別資料夾

# 類別名稱到數字的映射
# category_to_number = {
#     "Other": 0,
#     "Seatbelt": 1,
#     "Distract": 2,
#     "Phone": 3,
#     "Drowsy": 4,
#     "No_Seatbelt": 5
# }

# category_to_number = {
#     "laughing": 0,
#     "Drowsy": 1,
# }

category_to_number = {}
for idx, category in enumerate(categories):
    category_to_number[category] = idx

# 写入classInd.txt文件
with open('classRef.txt', 'w') as f:
    for category, idx in category_to_number.items():
        f.write(f"{idx} {category}\n")

with open('classInd.txt', 'w') as f:
    for idx in category_to_number.values():
        f.write(f"{idx}\n")

train_list = []
val_list = []
test_list = []

for category in categories:
    category_number = category_to_number[category]  # 獲取類別對應的數字
    category_path = os.path.join(base_path, category)
    videos = os.listdir(category_path)  # 列出該類別下的所有影片
    
    random.shuffle(videos)  # 對影片進行隨機排序
    
    num_videos = len(videos)
    num_train = int(num_videos * 0.75)  # 75% 作為訓練集
    num_val = int(num_videos * 0.95)  # 加上20%作為驗證集（總共95%）
    
    train_videos = videos[:num_train]
    val_videos = videos[num_train:num_val]
    test_videos = videos[num_val:]

    for video in train_videos:
        video_path = os.path.join(category_path, video)
        frame_count = get_frame_count(video_path)
        line = f"{video_path} {frame_count} {category_number}\n"
        train_list.append(line)
    
    for video in val_videos:
        video_path = os.path.join(category_path, video)
        frame_count = get_frame_count(video_path)
        line = f"{video_path} {frame_count} {category_number}\n"
        val_list.append(line)
        
    for video in test_videos:
        video_path = os.path.join(category_path, video)
        frame_count = get_frame_count(video_path)
        line = f"{video_path} {frame_count} {category_number}\n"
        test_list.append(line)

# 將訓練、驗證和測試列表寫入到對應的文件中
with open("train.txt", "w") as train_file:
    for line in train_list:
        train_file.write(line)

with open("val.txt", "w") as val_file:
    for line in val_list:
        val_file.write(line)

with open("test.txt", "w") as test_file:
    for line in test_list:
        test_file.write(line)

shutil.move("classRef.txt", new_path)
shutil.move("classInd.txt", new_path)
shutil.move("test.txt", new_path)
shutil.move("train.txt", new_path)
shutil.move("val.txt", new_path)
shutil.move("test.txt", new_path)