import os

def write_avi_paths(folder_path, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.avi'):
                    avi_path = os.path.join(root, file)
                    f.write(avi_path + '\n')

folder_path = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Drama/drama_video_new/"  # 指定文件夹的路径
output_file = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Drama/drama_video_label/drama.txt"  # 指定输出文件的路径

write_avi_paths(folder_path, output_file)
