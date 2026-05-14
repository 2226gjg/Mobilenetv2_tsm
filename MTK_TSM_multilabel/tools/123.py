# 假設 train_data 變數存放著 label.txt 的路徑
train_data = "/data/ivs01/MTK_TSM/mtk_dms_data_20240805_label/train.txt"

# 計算training時每個class數量的倒數，以用來平衡lr weights
class_count = {}

# 打開並讀取 label.txt 文件
with open(train_data, 'r') as file:
    for line in file:
        # 分割每一行，取最後一個元素作為類別
        class_label = int(line.strip().split()[-1])  # 確保類別標籤為整數
        
        # 更新字典中的類別數量
        if class_label in class_count:
            class_count[class_label] += 1
        else:
            class_count[class_label] = 1

# 確定最大類別 ID 以初始化 class_num 列表
max_class_id = max(class_count.keys())
class_num = [0] * (max_class_id + 1)

# 將字典中的數量放入對應的 class_num 列表位置
for class_id, count in class_count.items():
    class_num[class_id] = count

print(class_num)
class_num = [1/x if x != 0 else 0 for x in class_num]

# 找到 class_num 中的最小非零數字
min_value = min([x for x in class_num if x != 0])

# 計算將最小值調整為1所需的乘數
multiplier = 1 / min_value

# 將 class_num 中的每個數字乘以這個乘數
class_num = [x * multiplier for x in class_num]

weights = class_num

print(weights)
print(multiplier)