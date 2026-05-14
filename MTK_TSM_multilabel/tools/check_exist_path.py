import os

def check_paths_and_filter(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    valid_lines = []
    for line in lines:
        parts = line.split()
        if parts:
            dir_path = parts[0]
            if os.path.exists(dir_path):
                valid_lines.append(line)
            else:
                print(dir_path,"不存在")
    
    # with open(file_path, 'w') as file:
    #     for line in valid_lines:
    #         file.write(line)

def main():
    file_path = "/data/ivs01/MTK_TSM/mtk_dms_data_20240712_label/test.txt"  # 替換成你的txt檔案的路徑
    check_paths_and_filter(file_path)

if __name__ == '__main__':
    main()
