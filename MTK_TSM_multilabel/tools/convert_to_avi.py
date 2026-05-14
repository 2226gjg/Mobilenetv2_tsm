import os
import subprocess

def convert_videos_in_folder(input_folder, output_folder):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Check for video extensions (you can add more if needed)
        if filename.endswith(('.mp4', '.mkv', '.flv', '.mov')):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.avi')
            
            # Convert video using ffmpeg
            subprocess.call(['ffmpeg', '-i', input_file, '-b:v', '5M', output_file])


    print("Conversion completed!")

def convert_videos_in_directory(directory):
    # Iterate over all files and subdirectories in the directory
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            # Construct full paths for input and output folders
            input_folder = os.path.join(root, folder)

            # output_folder = os.path.join("/data/ivs01/MTK_TSM/mtk_dms_20240924/mtk_dms_data_20240924_new/",folder)
            output_folder = os.path.join("/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Drama/drama_video_new/",folder)
            # Convert videos in the current folder
            convert_videos_in_folder(input_folder, output_folder)

# Change this to your parent folder path
# parent_folder = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Training/"
parent_folder = "/data/ivs01/MTK_TSM/ivslab/Frank_tmp/video_for_MTK/Drama/drama_video/"
convert_videos_in_directory(parent_folder)
