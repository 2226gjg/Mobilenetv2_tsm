import cv2
import os

# 資料夾路徑
# folder_path = "/home/3105825/nova_action/output_frame_all/home/3105825/nova_action/mtk_dms_data_1219_drosy_laugh_new/Drowsy/qbox_0211_11_30_35/"
folder_path = "/home/3105825/nova_action/output_frame_all/home/3105825/nova_action/test_video_new/test/"
output_video_path = "/home/3105825/nova_action/output_video/mtk_dms_data_1219_drosy_laugh_new/qbox_0211_11_30_35.mp4"

output_directory = os.path.dirname(output_video_path)
# 检查目录是否存在
if not os.path.exists(output_directory):
    # 如果不存在，则创建目录
    os.makedirs(output_directory)

def sort_key(s):
    return int(s.split('frame')[1].split('.jpg')[0])

# 讀取資料夾中的所有文件名稱並排序
images = [img for img in os.listdir(folder_path) if img.startswith("frame")]
images.sort(key=sort_key)
print(images)
# 檢查是否有圖片
if len(images) == 0:
    print("No images found in the specified directory!")
    exit()

# 獲取第一張圖片的寬度和高度
frame = cv2.imread(os.path.join(folder_path, images[0]))
h, w, layers = frame.shape
size = (w, h)

# 定義編碼方式和創建VideoWriter物件
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

# 讀取資料夾中的所有圖片並寫入影片
for image in images:
    img_path = os.path.join(folder_path, image)
    img = cv2.imread(img_path)
    out.write(img)
    print(f"Writing frame {image}")

out.release()
print(f"Video {output_video_path} created successfully!")