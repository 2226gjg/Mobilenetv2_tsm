# 打開並讀取txt文件
with open("/data/ivs01/MTK_TSM/mtk_dms_data_20240805_label/val.txt", 'r') as file:
    lines = file.readlines()

# 初始化一個字典來存儲每個類別的計數
category_counts = {str(i): 0 for i in range(17)}

# 遍歷每一行，提取類別並更新計數
for line in lines:
    # 假設每行的類別在最後一個字符（去掉換行符）
    category = line.strip().split()[-1]
    if category in category_counts:
        category_counts[category] += 1

# 打印每個類別的行數
for category, count in category_counts.items():
    print(f'Category {category}: {count} lines')
