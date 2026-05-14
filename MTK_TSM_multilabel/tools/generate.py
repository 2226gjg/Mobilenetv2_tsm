import os
import argparse


parser = argparse.ArgumentParser()
# parser.add_argument('--datadir', help='Path to the all input data', type=str)
datadir = "/home/3105825/nova_action/tools/test/"
image_files = []
os.chdir(os.path.join(datadir))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".mp4") :
        image_files.append(datadir + filename)
os.chdir("..")
with open("/home/3105825/nova_action/tools/test/path.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")