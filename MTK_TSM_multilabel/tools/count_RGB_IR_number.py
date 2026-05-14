# import os
# import cv2
# import numpy as np

# def is_grayscale(video_path):
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # 將影像轉為灰階
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # 將灰階影像轉回BGR
#         gray_bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
#         # 計算原影像與灰階影像的差異
#         difference = cv2.absdiff(frame, gray_bgr_frame)
        
#         # 如果差異很小（影片基本是灰階的），就認為是黑白影片
#         if np.sum(difference) > 1000:  # 這個值可以根據實際情況調整
#             cap.release()
#             return False

#     cap.release()
#     return True

# def count_videos_from_list(file_path):
#     grayscale_count = 0
#     color_count = 0

#     # 從 TXT 檔案讀取影片路徑
#     with open(file_path, 'r') as file:
#         video_paths = [line.strip() for line in file.readlines()]

#     for video_path in video_paths:
#         if video_path.endswith(('.mp4', '.avi', '.mkv', '.mov')) and os.path.exists(video_path):
#             if is_grayscale(video_path):
#                 grayscale_count += 1
#             else:
#                 color_count += 1

#     return grayscale_count, color_count

# def main(txt_file):
#     grayscale_count, color_count = count_videos_from_list(txt_file)
#     total_count = grayscale_count + color_count
#     if total_count > 0:
#         print(f"Grayscale videos: {grayscale_count}, Percent: {grayscale_count / total_count * 100:.2f}%")
#         print(f"Color videos: {color_count}, Percent: {color_count / total_count * 100:.2f}%")
#     else:
#         print("No videos found or invalid file paths.")

# if __name__ == "__main__":
#     txt_file = "/data/ivs01/MTK_TSM/mtk_dms_20240905/mtk_dms_data_20240905_label/count_RGBIR.txt"  # 替換為實際的TXT檔案路徑
#     main(txt_file)


import os
import cv2
import numpy as np
from collections import defaultdict

def is_grayscale(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        difference = cv2.absdiff(frame, gray_bgr_frame)
        
        if np.sum(difference) > 1000:
            cap.release()
            return False

    cap.release()
    return True

def count_rgb_ir_from_list(file_path):
    folder_stats = defaultdict(lambda: {'RGB': 0, 'IR': 0})

    with open(file_path, 'r') as file:
        video_paths = [line.strip() for line in file.readlines()]

    for video_path in video_paths:
        if video_path.endswith(('.mp4', '.avi', '.mkv', '.mov')) and os.path.exists(video_path):
            # 假設資料夾名稱在影片路徑中，例如 /Distract 或其他資料夾
            folder_name = os.path.basename(os.path.dirname(video_path))

            if "IR" in video_path or "ir" in video_path:
                folder_stats[folder_name]['IR'] += 1
            else:
                if is_grayscale(video_path):
                    folder_stats[folder_name]['IR'] += 1  # 假設灰階影片是IR
                else:
                    folder_stats[folder_name]['RGB'] += 1

    return folder_stats

def main(txt_file):
    folder_stats = count_rgb_ir_from_list(txt_file)

    for folder_name, stats in folder_stats.items():
        total_videos = stats['RGB'] + stats['IR']
        if total_videos > 0:
            rgb_ratio = stats['RGB'] / total_videos * 100
            ir_ratio = stats['IR'] / total_videos * 100
            print(f"Folder '{folder_name}': RGB videos: {stats['RGB']}, IR videos: {stats['IR']}")
            print(f"RGB ratio: {rgb_ratio:.2f}%, IR ratio: {ir_ratio:.2f}%")
        else:
            print(f"Folder '{folder_name}' has no videos.")

if __name__ == "__main__":
    txt_file = "/data/ivs01/MTK_TSM/mtk_dms_20240924/mtk_dms_data_20240924_label/count_RGBIR.txt"  # 替換為實際的TXT檔案路徑
    main(txt_file)
