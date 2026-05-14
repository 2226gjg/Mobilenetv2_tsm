import cv2
import os
import argparse
import re

# 自然排序的 key 函數
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def merge_videos_from_folder(folder_path):
    # 獲取所有子資料夾及主資料夾中的 MP4 影片檔案名稱，並按自然順序排序
    video_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.mp4'):
                video_files.append(os.path.join(root, f))
    
    # 按檔案名稱自然排序
    video_files = sorted(video_files, key=natural_sort_key)
    
    if len(video_files) == 0:
        print("資料夾中沒有找到任何 MP4 影片")
        return
    
    # 初始化影片列表
    videos = []
    
    # 開啟所有影片並獲取影片的基本參數（例如寬度、高度和 FPS）
    for path in video_files:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"無法開啟影片: {path}")
            continue
        videos.append(cap)
    
    if len(videos) == 0:
        print("沒有可以合併的影片")
        return
    
    # 取第一個影片的寬度、高度和 FPS 來設置輸出的影片格式
    width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(videos[0].get(cv2.CAP_PROP_FPS))
    
    # 設置輸出影片格式和檔案路徑（輸出位置為母資料夾）
    output_path = os.path.join(folder_path, 'merged_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定 MP4 編碼格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 開始合併每部影片
    for cap in videos:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    # 釋放資源
    out.release()
    print(f"合併完成，輸出路徑為: {output_path}")

if __name__ == "__main__":
    # 設定 argparse 來解析資料夾路徑
    parser = argparse.ArgumentParser(description="合併母資料夾下所有子資料夾內的 MP4 影片")
    parser.add_argument('folder', type=str, help='包含 MP4 影片的母資料夾路徑')
    
    # 解析輸入參數
    args = parser.parse_args()
    
    # 合併資料夾中的所有 MP4 影片
    merge_videos_from_folder(args.folder)
