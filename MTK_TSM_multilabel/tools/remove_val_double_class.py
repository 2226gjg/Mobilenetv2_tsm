import cv2
import os
import shutil

def compare_videos(video_path1, video_path2):
    # 確保文件存在
    if not os.path.exists(video_path1):
        print(f"文件不存在: {video_path1}")
        return False
    if not os.path.exists(video_path2):
        print(f"文件不存在: {video_path2}")
        return False

    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # 檢查視頻是否打開成功
    if not cap1.isOpened():
        print(f"無法打開視頻文件: {video_path1}")
        return False
    if not cap2.isOpened():
        print(f"無法打開視頻文件: {video_path2}")
        return False

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        if frame1.shape != frame2.shape or not (frame1 == frame2).all():
            cap1.release()
            cap2.release()
            return False

    cap1.release()
    cap2.release()
    return True

def remove_frame_from_path(path):
    # 移除包含 "_frame" 的部分
    parts = path.split('/')
    new_parts = [part for part in parts if '_frame' not in part]
    return '/'.join(new_parts)

def copy_to_folder(source_path, destination_folder):
    # 檢查源路徑是否存在
    if os.path.exists(source_path+'.mp4'):
        # 確保目標資料夾存在，如果不存在，則創建它
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        # 構建目標路徑
        destination_path = os.path.join(destination_folder, os.path.basename(source_path)+'.mp4')
        
        # 複製文件
        shutil.copy2(source_path+'.mp4', destination_path)
        print(f"已將 {source_path} 複製到 {destination_path}")
    else:
        print(f"錯誤: 路徑 {source_path} 不存在")


def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 打印處理前的行數
    print(f"處理前的行數: {len(lines)}")

    # 分割行成組件
    data = [line.strip().split(' ') for line in lines]

    # 提取包含 'Drowsy' 或 'Phone' 的路徑及其對應的行
    drowsy_paths = [(index, line[0]) for index, line in enumerate(data) if any('Drowsy' in part for part in line[0].split('/'))]
    phone_paths = [(index, line[0]) for index, line in enumerate(data) if any('Phone' in part for part in line[0].split('/'))]
    neg_paths = [(index, line[0]) for index, line in enumerate(data) if any('negative' in part for part in line[0].split('/'))]
    # 移除路徑中包含 "_frame" 的部分
    drowsy_paths = [(index, remove_frame_from_path(path)) for index, path in drowsy_paths]
    phone_paths = [(index, remove_frame_from_path(path)) for index, path in phone_paths]
    neg_paths = [(index, remove_frame_from_path(path)) for index, path in neg_paths]

    print(drowsy_paths)
    print(phone_paths)
    count = 0

    # for d_index, d_path in drowsy_paths:
    #     copy_to_folder(d_path,"/home/3105825/nova_action/tools/test/")
    # for p_index, p_path in phone_paths:
    #     copy_to_folder(p_path,"/home/3105825/nova_action/tools/test/")

    for n_index, n_path in neg_paths:
        copy_to_folder(n_path,"/home/3105825/nova_action/tools/test/")






    # # 比較影片並找出需要移除的索引
    # to_remove_indices = set()
    # for d_index, d_path in drowsy_paths:
    #     for p_index, p_path in phone_paths:
    #         if os.path.exists(d_path+ '.mp4') and os.path.exists(p_path+ '.mp4'):
    #             print(count)
    #             count = count + 1
    #             if compare_videos(d_path+ '.mp4', p_path+ '.mp4'):
    #                 to_remove_indices.update([d_index, p_index])
    # # 打印要移除的索引以進行調試
    # print(f"要移除的索引: {to_remove_indices}")

    # # 移除具有相同文件夾名稱的行
    # new_data = [line for index, line in enumerate(lines) if index not in to_remove_indices]

    # # 打印處理後的行數
    # print(f"處理後的行數: {len(new_data)}")

    # # 將結果寫回文件
    # with open(file_path, 'w', encoding='utf-8') as file:
    #     file.writelines(new_data)

    # # 打印處理前後的行數
    # print(f"處理前的行數: {len(lines)}")
    # print(f"處理後的行數: {len(new_data)}")






# 處理文件
process_file("/home/3105825/nova_action/mtk_dms_data_20240409_label/val.txt")
