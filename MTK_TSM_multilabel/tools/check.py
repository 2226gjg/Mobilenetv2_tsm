def read_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return lines

def main():
    # 讀取文件a.txt和b.txt的所有行
    lines_a = read_lines("/home/3105825/nova_action/mtk_dms_data_20240617_label/train.txt")
    lines_b = read_lines("/home/3105825/nova_action/mtk_dms_data_20240617_label/val.txt")
    
    # 將兩個文件的行轉換為集合
    set_a = set(lines_a)
    set_b = set(lines_b)
    
    # 計算兩個集合的交集
    intersection_set = set_a.intersection(set_b)
    
    # 判斷交集是否為空
    if len(intersection_set) == 0:
        print("文件a.txt和b.txt的交集為0")
    else:
        print("文件a.txt和b.txt的交集不為0，具體交集內容如下：")
        for line in intersection_set:
            print(line)

if __name__ == "__main__":
    main()
