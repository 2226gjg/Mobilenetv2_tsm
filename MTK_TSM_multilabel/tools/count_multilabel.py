import os

# 定義類別順序
class_names = [
    'no_hand_driving', 'nap', 'seatbelt', 'Distract', 'Phone', 'Drowsy',
    'laughing', 'single_hand_driving', 'negative', 'Singing',
    'Sneezing', 'Eyescratching', 'Talking', 'Eating', 'Smoking'
]

# 初始化每個類別的計數器
class_counts = {class_name: 0 for class_name in class_names}

# 讀取檔案
file_path = "/data/ivs01/MTK_TSM/mtk_dms_20240924_multilabel/mtk_dms_data_20240924_multilabel_label/all_multilabel.txt"
with open(file_path, 'r') as f:
    for line in f:
        # 分割每行的數據
        parts = line.strip().split()
        # 取得多標籤的數據（去掉方括號並轉換成整數列表）
        label_str = parts[-1].strip('[]')
        labels = [int(x) for x in label_str.split(',')]

        # 更新每個類別的計數
        for i, label in enumerate(labels):
            if label == 1:
                class_counts[class_names[i]] += 1

# 輸出每個類別的計數
for class_name, count in class_counts.items():
    print(f'{class_name}: {count}')
