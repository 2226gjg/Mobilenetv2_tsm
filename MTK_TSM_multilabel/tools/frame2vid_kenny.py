import cv2
import os
import argparse

def sort_key(s):
    return int(s.split('frame')[1].split('.jpg')[0])

def main():
    parser = argparse.ArgumentParser(description="Process file path.")
    parser.add_argument('--folderpath', type=str, required=True, help='')
    parser.add_argument('--outputpath', type=str, required=True, help='')

    args = parser.parse_args()

    folder_path = args.folderpath
    output_directory = args.outputpath


    # 检查目录是否存在
    if not os.path.exists(output_directory):
        # 如果不存在，则创建目录
        os.makedirs(output_directory)

    # 尋找所有類別資料夾
    for cls in os.listdir(folder_path):
        cls_path = os.path.join(folder_path, cls)
        if not os.path.isdir(cls_path):
            continue
        
        # 尋找每個類別資料夾中的檔案
        for name in os.listdir(cls_path):
            name_path = os.path.join(cls_path, name)
            if not os.path.isdir(name_path):
                continue

            # 讀取資料夾中的所有文件名稱並排序
            images = [img for img in os.listdir(name_path) if img.startswith("frame")]
            images.sort(key=sort_key)

            # 檢查是否有圖片
            if len(images) == 0:
                print(f"No images found in {name_path}!")
                continue
            
            print(images)

            # 獲取第一張圖片的寬度和高度
            frame = cv2.imread(os.path.join(name_path, images[0]))
            h, w, layers = frame.shape
            size = (w, h)

            out_cls_dir = os.path.join(output_directory, cls)
            if not os.path.exists(out_cls_dir):
                # 如果不存在，则创建目录
                os.makedirs(out_cls_dir)

            # 定義編碼方式和創建VideoWriter物件
            output_video_path = os.path.join(output_directory, cls, f"{name}.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
            
            # 讀取資料夾中的所有圖片並寫入影片
            for image in images:
                img_path = os.path.join(name_path, image)
                img = cv2.imread(img_path)
                out.write(img)
                print(f"Writing frame {image}")
            
            # 釋放VideoWriter物件
            out.release()
            
            print(f"Video {output_video_path} created successfully!")

if __name__ == "__main__":
    main()