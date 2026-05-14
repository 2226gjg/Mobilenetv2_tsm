import os
import time
from ops.models import TSN
from ops.transforms import *
import cv2
from PIL import Image
import torch
import numpy as np
from thop import clever_format
from thop import profile
from torchsummary import summary
import torch.nn as nn
import argparse
'''
Inference code for TSM action recognition model

input 8 image sequentially and produce one prediction class

first inference: frame_num 1~8 ==> pred
second inference: frame_num 2~9 ==> pred
third inference: frame_num 3~10 ==> pred

check list:
    correct model ?
    correct input ?
    correct input mean and std ?



'''
def main():
    parser = argparse.ArgumentParser(description="Process file path.")
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test file')
    parser.add_argument('--resume', type=str, required=True, help='Path to the model')
    parser.add_argument('--refpath', type=str, required=True, help='Path to the ref label')
    parser.add_argument('--outpath', type=str, required=True, help='Path to the ref label')
    parser.add_argument('--outputpath', type=str, required=True, help='Path to the ref label')


    args = parser.parse_args()

    test_file_path = args.test_path
    resume = args.resume
    Ref_path = args.refpath
    out_path = args.outpath
    outputpath = args.outputpath



    # Ref_path = "/data/ivs01/MTK_TSM/mtk_dms_data_20240617_label/classRef.txt"
    with open(Ref_path, 'r') as file:  #從ClassInd讀class
        lines = file.readlines()
    class_num = len(lines) - 1
    cls_text = []
    for line in lines:
        # 去除每行末尾的换行符并添加到列表中
        cls_text.append(line.strip().split(' ', 1)[1])
    print(cls_text)


    # arch = 'resnet50'
    arch = 'mobilenetv2'
    # arch = 'mobilenetv4'
    # num_class = 3
    num_class = len(lines)
    num_segments = 8
    modality = 'RGB'
    # modality = 'YUV'
    # base_model = 'resnet50'
    base_model = 'mobilenetv2'
    # base_model = 'mobilenetv4'
    consensus_type='avg'
    dataset = 'ucf101'
    dropout = 0.9
    img_feature_dim = 256
    no_partialbn = True
    pretrain = 'imagenet'
    shift = True
    shift_div = 8
    shift_place = 'blockres'
    temporal_pool = False
    non_local = False
    tune_from = None
    device_available = None
    motion_only = True
    edge_dataset = False
    number_of_skip_frame=1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("###################device ####################= ",device)

    #load model
    model = TSN(num_class, num_segments, modality,
                    base_model=arch,
                    consensus_type=consensus_type,
                    dropout=dropout,
                    img_feature_dim=img_feature_dim,
                    partial_bn=not no_partialbn,
                    pretrain=pretrain,
                    is_shift=shift, shift_div=shift_div, shift_place=shift_place,
                    fc_lr5=not (tune_from and dataset in tune_from),
                    temporal_pool=temporal_pool,
                    non_local=non_local)

    model = torch.nn.DataParallel(model).to(device)
    
    # resume = "/home/s310511029/novatek_behaviour_recognition_tsm/temporal-shift-module/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e50_cos_20230308_75/ckpt.best.pth.tar" #  the last weights
    # resume = "/home/s310511029/novatek_behaviour_recognition_tsm/temporal-shift-module/checkpoint/TSM_ucf101_YUV_resnet50_shift8_blockres_avg_segment8_e100_cos/ckpt.best.pth.tar"
    # if motion_only:
        # resume = "/data/ivs01/MTK_TSM/checkpoint/imagenet_mean_std_20240617_cos_cos/ckpt.best.pth.tar"
    # else:
        # resume = "/data/ivs01/MTK_TSM/checkpoint/imagenet_mean_std_20240617_cos_cos/ckpt.best.pth.tar"
        


    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)



    # print('###################################')


    new_fc_state_dict = model.module.new_fc.state_dict()
    out_ch = new_fc_state_dict['weight'].shape[0]
    in_ch = new_fc_state_dict['weight'].shape[1]
    New_output_Conv = nn.Conv2d(in_ch, out_ch, 1, 1)
    New_output_Conv.load_state_dict({"weight":new_fc_state_dict["weight"].view(out_ch, in_ch, 1, 1),
                            "bias":new_fc_state_dict["bias"]})

    heatmap_output_layer = New_output_Conv.to(device)

    # model.module.base_model.avgpool = nn.Identity()
    # model.module.base_model.fc = nn.Identity()
    # model.module.new_fc = nn.Identity()
    # model.module.consensus = nn.Identity()
    # print(model)


    # input = torch.randn(1,24,256,256).to(device)
    # output_test, output_test2 = model(input)


    # print(output_test.shape)
    # print(output_test2.shape)
    # output_test = output_test.view(-1, 2048, 8, 8)
    # layer = New_output_Conv.to(device)
    # test_output2 = layer(output_test)


    # # print(model)
    # print(test_output2.size())
    # if consensus_type =='avg':
    #     test_output2 = test_output2.mean(dim=0, keepdim=True)
    # else:
    #     test_output2 = test_output2

    # print(test_output2.size())
    # print('###################################')



    #how to deal with the pictures
    # 128 mvx mvy
    if motion_only:
        # input_mean = [0.501, 0.502, 0.502]
        # input_std =  [0.229, 0.013, 0.031]
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]
    else:
    # gray gray gray
    #    input_mean =  [0.419, 0.419, 0.419]
    #    input_std =   [0.232, 0.232, 0.232]

    #gray motion
        # input_mean = [0.419, 0.502, 0.502]
        # input_std =  [0.232, 0.013, 0.031]

    #edge motion
        # input_mean = [0.042, 0.502, 0.502]
        # input_std = [0.077, 0.012, 0.014]  

    #imagenet
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]


    crop_size = 256
    scale_size =256 * 256 // 224
    normalize = GroupNormalize(input_mean, input_std)
    transform = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
        normalize,
    ])

    # video_path = "/home/s310511029/novatek_behaviour_recognition_tsm/temporal-shift-module/test_video/4_class/brake.avi"
    # video_path = '/home/s310511029/dataset/behaviour_dataset/classification_4_class_20230320_gray_motion/other/v_other_g609_c04.avi'



    video_file_list = []
    folder_list = []
    label = []
    if edge_dataset:
        dir_full_path = './dataset/behaviour_dataset/classification_4_class_20230430_edge_motion/'
    else:
        dir_full_path = ''
    # test_file_path = "/data/ivs01/MTK_TSM/test_video_label/test_laughing.txt"
    # test_file_path = "/data/ivs01/MTK_TSM/mtk_dms_data_20240617_label/test.txt"
    counter = 0
    f = open(test_file_path)
    for line in f.readlines():

        line = line.strip('\n')
        video_file_list.append(line)
        # line = line.strip('.avi')
        # folder_list.append(line)    
        # if '/Distract/' in line:
        #     label.append(0)
        # elif '/Phone/' in line:
        #     label.append(1)
        # elif '/Drowsy/' in line:
        #     label.append(2)
        # # elif 'other' in line:
        # #     label.append(3)
        # else:
        #     pass
        
    
        # counter+=1
    f.close

    print("video_file_list = ")
    # print(video_file_list)
    # print("label = ")
    # print(label)
    # print("folder_list = ")
    # print(folder_list)



    # cls_text = ['crossing','cutin','emergency_brake','no behavior']
    # cls_color = [(0,255,0),(0,0,255),(255,0,0),(255,255,255)]
    # cls_text = ['Distract','Phone','Drowsy','no behavior']
    # cls_color = [(0,255,0),(0,0,255),(255,0,0),(255,255,255)]


    # cls_color = [(255,255,255),(128,0,128),(0,255,0),(0,0,255),(255,0,0),(128,128,128)]
    # cls_text = ['no_hand_driving','nap','seatbelt','Distract','Phone','Drowsy','laughing','single_hand_driving','negative','no_seatbelt']
    # Ref_path = "/data/ivs01/MTK_TSM/mtk_dms_data_20240617_label/classRef.txt"
    # with open(Ref_path, 'r') as file:  #從ClassInd讀class
    #     lines = file.readlines()
    # class_num = len(lines) - 1
    # cls_text = []
    # for line in lines:
    #     # 去除每行末尾的换行符并添加到列表中
    #     cls_text.append(line.strip().split(' ', 1)[1])
    # print(cls_text)

    cls_color = [
        (255, 255, 255),  # White
        (178, 0, 178),    # Lighter Purple
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 51, 51),    # Lighter Red
        (128, 128, 128),  # Grey
        (255, 255, 0),    # Yellow
        (255, 165, 0),    # Orange
        (0, 255, 255),    # Cyan
        (255, 192, 203),  # Pink
        (144, 238, 144),  # Light Green
        (255, 182, 193),  # Light Pink
        (173, 216, 230),  # Light Blue
        (240, 230, 140),  # Khaki
        (135, 206, 250),  # Light Sky Blue
        (211, 211, 211)   # Light Gray
    ]


    # cls_text = ['laughing','Drowsy']
    # cls_color = [(255,0,0),(0,255,0)]


    icon_distract = cv2.imread('/data/ivs01/MTK_TSM/icon/distract.jpg')
    icon_distract_resized = cv2.resize(icon_distract, (50, 50))  # 假设我们想要50x50尺寸的图标
    
    icon_drowsy = cv2.imread('/data/ivs01/MTK_TSM/icon/drowsy.jpg')
    icon_drowsy_resized = cv2.resize(icon_drowsy, (50, 50))  # 假设我们想要50x50尺寸的图标
    
    icon_phone = cv2.imread('/data/ivs01/MTK_TSM/icon/phone.jpg')
    icon_phone_resized = cv2.resize(icon_phone, (50, 50))  # 假设我们想要50x50尺寸的图标
    
    icon_seatbelt = cv2.imread('/data/ivs01/MTK_TSM/icon/seatbelt.jpg')
    icon_seatbelt_resized = cv2.resize(icon_seatbelt, (50, 50))  # 假设我们想要50x50尺寸的图标
    
    icon_laughing = cv2.imread('/data/ivs01/MTK_TSM/icon/laughing.jpg')
    icon_laughing_resized = cv2.resize(icon_laughing, (50, 50))  # 假设我们想要50x50尺寸的图标
    
    icon_steer_wheel = cv2.imread('/data/ivs01/MTK_TSM/icon/steer_wheel.jpg')
    icon_steer_wheel_resized = cv2.resize(icon_steer_wheel, (50, 50))  # 假设我们想要50x50尺寸的图标

    icon_singing = cv2.imread('/data/ivs01/MTK_TSM/icon/singing.jpg')
    icon_singing_resized = cv2.resize(icon_singing, (50, 50))  # 假设我们想要50x50尺寸的图标

    icon_sneezing = cv2.imread('/data/ivs01/MTK_TSM/icon/sneezing.jpg')
    icon_sneezing_resized = cv2.resize(icon_sneezing, (50, 50))  # 假设我们想要50x50尺寸的图标

    icon_eyescratching = cv2.imread('/data/ivs01/MTK_TSM/icon/eyescratching.jpg')
    icon_eyescratching_resized = cv2.resize(icon_eyescratching, (50, 50))  # 假设我们想要50x50尺寸的图标

    icon_talking = cv2.imread('/data/ivs01/MTK_TSM/icon/talking.jpg')
    icon_talking_resized = cv2.resize(icon_talking, (50, 50))  # 假设我们想要50x50尺寸的图标

    icon_smoking = cv2.imread('/data/ivs01/MTK_TSM/icon/smoking.jpg')
    icon_smoking_resized = cv2.resize(icon_smoking, (50, 50))  # 假设我们想要50x50尺寸的图标

    icon_eating = cv2.imread('/data/ivs01/MTK_TSM/icon/eating.jpg')
    icon_eating_resized = cv2.resize(icon_eating, (50, 50))  # 假设我们想要50x50尺寸的图标
    
    if icon_distract is None:
        print("Failed to load icon from:")

    action_counter = 0
    for idxx, video_path in enumerate(video_file_list):
        # folder = video_path.strip('.avi')

        # 獲取文件的基本名稱，並去除擴展名
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # 獲取文件所在的路徑
        dir_path = os.path.dirname(video_path)

        # 獲取上一層文件夾的名稱
        parent_folder = os.path.basename(dir_path)
        print(parent_folder)
        # 結合上一層文件夾的名稱和文件的基本名稱
        result = os.path.join('/', parent_folder, base_name)
        
        print("base_name:",base_name)



        os.makedirs(out_path + result+'/', exist_ok=True)
        print(result)
        pil_img_list = list()
        print(dir_full_path+video_path)
        cap = cv2.VideoCapture(dir_full_path+video_path) 
        start_time = time.time()
        counter = 0
        frame_numbers = 0
        training_fps = 30
        fps_calculation_latest = training_fps
        training_time = 2
        inference_flag = False #False ==> wait until len of pil_img_list == 8 ==> flag turn into True   
        fps = cap.get(cv2.CAP_PROP_FPS) 

        if fps < 1:
            fps = 30
            
        duaring = int(fps * training_time / num_segments)   
        state = 0
        t1 = time.time()
        
        
        icon_distract = cv2.imread('/data/ivs01/MTK_TSM/icon/distract.jpg')
        icon_distract_resized = cv2.resize(icon_distract, (50, 50))  # 假设我们想要50x50尺寸的图标
        
        icon_drowsy = cv2.imread('/data/ivs01/MTK_TSM/icon/drowsy.jpg')
        icon_drowsy_resized = cv2.resize(icon_drowsy, (50, 50))  # 假设我们想要50x50尺寸的图标
        
        icon_phone = cv2.imread('/data/ivs01/MTK_TSM/icon/phone.jpg')
        icon_phone_resized = cv2.resize(icon_phone, (50, 50))  # 假设我们想要50x50尺寸的图标
        
        icon_seatbelt = cv2.imread('/data/ivs01/MTK_TSM/icon/seatbelt.jpg')
        icon_seatbelt_resized = cv2.resize(icon_seatbelt, (50, 50))  # 假设我们想要50x50尺寸的图标
        
        icon_laughing = cv2.imread('/data/ivs01/MTK_TSM/icon/laughing.jpg')
        icon_laughing_resized = cv2.resize(icon_laughing, (50, 50))  # 假设我们想要50x50尺寸的图标
        
        icon_steer_wheel = cv2.imread('/data/ivs01/MTK_TSM/icon/steer_wheel.jpg')
        icon_steer_wheel_resized = cv2.resize(icon_steer_wheel, (50, 50))  # 假设我们想要50x50尺寸的图标

        icon_singing = cv2.imread('/data/ivs01/MTK_TSM/icon/singing.jpg')
        icon_singing_resized = cv2.resize(icon_singing, (50, 50))  # 假设我们想要50x50尺寸的图标

        icon_sneezing = cv2.imread('/data/ivs01/MTK_TSM/icon/sneezing.jpg')
        icon_sneezing_resized = cv2.resize(icon_sneezing, (50, 50))  # 假设我们想要50x50尺寸的图标

        icon_eyescratching = cv2.imread('/data/ivs01/MTK_TSM/icon/eyescratching.jpg')
        icon_eyescratching_resized = cv2.resize(icon_eyescratching, (50, 50))  # 假设我们想要50x50尺寸的图标

        icon_talking = cv2.imread('/data/ivs01/MTK_TSM/icon/talking.jpg')
        icon_talking_resized = cv2.resize(icon_talking, (50, 50))  # 假设我们想要50x50尺寸的图标

        icon_smoking = cv2.imread('/data/ivs01/MTK_TSM/icon/smoking.jpg')
        icon_smoking_resized = cv2.resize(icon_smoking, (50, 50))  # 假设我们想要50x50尺寸的图标

        icon_eating = cv2.imread('/data/ivs01/MTK_TSM/icon/eating.jpg')
        icon_eating_resized = cv2.resize(icon_eating, (50, 50))  # 假设我们想要50x50尺寸的图标
        
        if icon_distract is None:
            print("Failed to load icon from:")

        fps_real = 30
        fps_nap = 50
        fps_sneezing = 8
        buffer_distract = [0] * 1 * fps_real
        buffer_phone = [0] * 2 * fps_real
        buffer_drowsy = [0] * 2 * fps_real
        buffer_nap = [0] * 1 * fps_nap
        buffer_laughing = [0] * 2 * fps_real
        buffer_seatbelt = [0] * 10 * fps_real
        buffer_steer_wheel = [0] * 2 * fps_real  
        buffer_singing = [0] * 2 * fps_real
        buffer_sneezing = [0] * 1 * fps_sneezing
        buffer_eyescratching = [0] * 2 * fps_nap
        buffer_talking = [0] * 2 * fps_real
        buffer_smoking = [0] * 2 * fps_real
        buffer_eating = [0] * 2 * fps_real  
        buffer_negative = [0] * 7 * fps_real  

        buffer_long_distraction = [0] * 3 * fps_real
        buffer_short_distraction = [0] * 30 * fps_real
        buffer_attention = [0] * 2 * fps_real
        buffer_drowsy_KSS = [0] * 60 * fps_real
        buffer_drowsy_KSS_flag = [0] * 2
        buffer_nap_KSS = [0] * 60 * fps_real
        buffer_nap_KSS_flag = [0] * 2


        play_icon_distract = [0] * 1 * fps_real  #讓條件達成至少icon提示1秒
        play_icon_phone = [0] * 1 * fps_real  
        play_icon_drowsy = [0] * 1 * fps_real
        play_icon_nap = [0] * 1 * fps_real
        play_icon_laughing = [0] * 1 * fps_real
        play_icon_seatbelt = [0] * 1 * fps_real
        play_icon_steer_wheel = [0] * 1 * fps_real  
        play_icon_singing = [0] * 1 * fps_real  
        play_icon_sneezing = [0] * 1 * fps_real
        play_icon_eyescratching = [0] * 1 * fps_real
        play_icon_talking = [0] * 1 * fps_real
        play_icon_smoking = [0] * 1 * fps_real  
        play_icon_eating = [0] * 1 * fps_real  
            
        icon_x_1 = 544 - 60  # 右半行icon
        icon_x_2 = 544 - 120  #左半行icon
        icon_y_start = 20  # 圖標的初始 Y 座標，從上往下出現
        icon_spacing = 10  # 圖標之間的間距
        
        # 要調整的目標解析度
        # target_width = 1920
        # target_height = 1080

        # 計算原始視頻的解析度
        # original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        while cap.isOpened():
            #############################################################################
            # read_img_time_start = time.time()
            #############################################################################
            ret, frame = cap.read()
            if ret:

                # frame = cv2.resize(frame, (target_width, target_height),interpolation=cv2.INTER_CUBIC)

                frame_numbers+=1
                print(frame_numbers)
                # print(len(pil_img_list))


                # if frame_numbers%duaring == 0 and len(pil_img_list)<8:
                #     frame_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                #     pil_img_list.extend([frame_pil])
                # if frame_numbers%duaring == 0 and  len(pil_img_list)==8:
                #     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                #     pil_img_list.pop(0)
                #     pil_img_list.extend([frame_pil])
                

                if len(pil_img_list) < num_segments and (frame_numbers%number_of_skip_frame==0):
                    if modality =='YUV':
                        # frame_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2YUV))

                        frame_pil = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
                        grey, img_mvx , img_mvy  = cv2.split(frame_pil)                  

                        if motion_only:
                            img_128 = np.empty([480, 640], dtype=np.uint8)
                            img_128.fill(128)
                            image_merged_motion = cv2.merge([img_128, img_mvx, img_mvy])
                            
                            frame_pil = Image.fromarray(image_merged_motion)
                        else:
                            #gray gray gray
                            # image_merged_grey = cv2.merge([grey, grey, grey])
                            # frame_pil = Image.fromarray(image_merged_grey)

                            #gray motion
                            image_merged = cv2.merge([grey, img_mvx, img_mvy])
                            frame_pil = Image.fromarray(image_merged)

                    else:
                        frame_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    pil_img_list.extend([frame_pil])

                if len(pil_img_list) == num_segments and (frame_numbers%number_of_skip_frame==0):
                    if modality =='YUV':
                        # frame_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2YUV))
                        #print(' ########################################################################time: {}'.format(time.time()-read_img_time_start))
                        frame_pil = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
                        #print(' #####################################################time: {}'.format(time.time()-read_img_time_start))                   
                        grey, img_mvx , img_mvy  = cv2.split(frame_pil)
                        #print(' ######################################time: {}'.format(time.time()-read_img_time_start))
                        if motion_only:
                            img_128 = np.empty([480, 640], dtype=np.uint8)
                            img_128.fill(128)
                            image_merged_motion = cv2.merge([img_128, img_mvx, img_mvy])
                            
                            frame_pil = Image.fromarray(image_merged_motion)
                        else:
                            #gray gray gray
                            # image_merged_grey = cv2.merge([grey, grey, grey])
                            # frame_pil = Image.fromarray(image_merged_grey)

                            #gray motion
                            image_merged = cv2.merge([grey, img_mvx, img_mvy])
                            frame_pil = Image.fromarray(image_merged)


                    else:
                        frame_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


                    #print(' #################time: {}'.format(time.time()-read_img_time_start))
                    pil_img_list.pop(0)
                    pil_img_list.extend([frame_pil])


                    #print(' #########time: {}'.format(time.time()-read_img_time_start))

                    inference_flag = True

                    input = transform(pil_img_list)
                    input = input.unsqueeze(0).to(device)


                    #############################################################
                    #read_img_time_end = time.time()
                    #print(' ###time: {}'.format(time.time()-read_img_time_start))
                    #print('image stack preparing time: {}'.format(read_img_time_end-read_img_time_start))  
                    ##############################################################

                    # print('input.shape:', str(input.shape))
                    input_for_heatmap, base_out, out = model(input)
                    print(out)
                    # out = torch.sigmoid(out)
                    # print('input_for_heatmap.shape: {}'.format(input_for_heatmap.shape))

                    ##############################################################
                    #model_time = time.time()-read_img_time_end
                    #print('model time: {}'.format(model_time))
                    #after_model_time_start = time.time()
                    ##############################################################

                    print(out.shape)
                    out = out.squeeze()
                    result_list = out.tolist()
                    # greater_than_zero = out > 1
                    # greater_than_zero = out 
                    # result_list = greater_than_zero.tolist()
                    print(result_list)

                    heatmap = heatmap_output_layer(input_for_heatmap)
                    # print('heatmap shape: {}'.format(heatmap.shape))

                    # print('base_out:{}'.format(base_out))

                    heatmap_outputs = []
                    heatmap_threshold = 0

                    for state, condition in enumerate(result_list):
                        # 如果對應的索引值為True，則正常提取heatmap_output
                        if condition:
                            heatmap_output = heatmap[-1, state, :, :].cpu().detach().numpy()
                            
                            # 應用閾值
                            heatmap_output[heatmap_output < heatmap_threshold] = 0
                            heatmap_output[heatmap_output > 255] = 255
                            
                            # 正規化和調整大小
                            heatmap_output = cv2.normalize(heatmap_output, None, 0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            heatmap_output = cv2.resize(heatmap_output, dsize=(frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                        else:
                            # 如果對應的索引值為False，則將heatmap_output設置為0
                            heatmap_output = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)  # 假設frame是二維的

                        heatmap_outputs.append(heatmap_output)  # 將結果添加到列表中
                    # print("heatmap_outputs",heatmap_outputs)
    
                    heatmap_visualization = np.zeros_like(frame)

                    # if state == 4:
                    #     heatmap_visualization[:,:,0] = heatmap_outputs
                    #     heatmap_visualization[:,:,1].fill(0)
                    #     heatmap_visualization[:,:,2].fill(0)
                    # elif state == 6:
                    #     heatmap_visualization[:,:,0] = heatmap_output
                    #     heatmap_visualization[:,:,1] = heatmap_output
                    #     heatmap_visualization[:,:,2].fill(0)
                    # elif state == 10:
                    #     heatmap_visualization[:,:,0] = heatmap_output
                    #     heatmap_visualization[:,:,1].fill(0)
                    #     heatmap_visualization[:,:,2].fill(0)
                    # elif state == 11:
                    #     heatmap_visualization[:,:,0] = heatmap_output
                    #     heatmap_visualization[:,:,1] = heatmap_output
                    #     heatmap_visualization[:,:,2].fill(0)
                    # elif state == 12:
                    #     heatmap_visualization[:,:,0].fill(0)
                    #     heatmap_visualization[:,:,1] = heatmap_output
                    #     heatmap_visualization[:,:,2] = heatmap_output
                    # elif state == 13:
                    #     heatmap_visualization[:,:,0].fill(0)
                    #     heatmap_visualization[:,:,1].fill(0)
                    #     heatmap_visualization[:,:,2] = heatmap_output[13]  
                    # print("heatmap_visualization",heatmap_visualization)


                    heatmap_visualization[:,:,0] = heatmap_outputs[6] #laughing B
                    heatmap_visualization[:,:,1] = heatmap_outputs[13] #talking G
                    heatmap_visualization[:,:,2] = heatmap_outputs[10] #singing R
                                
                    print('==============================================')


                key = cv2.waitKey(1) & 0xff
                if key == ord(" "):
                    cv2.waitKey(0)
                if key == ord("q"):
                    break
                counter += 1
                if counter%number_of_skip_frame==0:
                    fps_calculation = counter / (time.time() - start_time)
                    fps_calculation_latest = fps_calculation               
                else:
                    fps_calculation = fps_calculation_latest 

                
                
                    
                if (time.time() - start_time) != 0 :
                    # frame = cv2.resize(frame, dsize=(320,240), interpolation=cv2.INTER_CUBIC)
                    if inference_flag == True:
                        # frame = cv2.addWeighted(frame, 0.7, heatmap_visualization, 0.3, 30)  #----------------------------------放heat_map-----------------------------
                        # frame = cv2.addWeighted(frame, 1-alpha, heatmap_colored.astype(np.uint8), alpha, 0)
                        # cv2.putText(frame, "pred: {0} {1} FPS".format((cls_text[state]), float('%.1f' % (fps_calculation))), (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, cls_color[state],3)
                        # print(state)
                        
                        # ---------------------------------------------------- 放字 ------------------------------------------------------------------------------------
                        # for state, condition in enumerate(result_list):
                        #     # display_text = "0" if not condition else "{:.2f}".format(out[state].item())
                        #     display_text = "{:.2f}".format(out[state].item())
                        #     text = "{} : {}".format(cls_text[state], display_text)
                        #     color = cls_color[state]



                        #     # 在 frame 上绘制文本
                        #     cv2.putText(frame, text, (10, 30 + state * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        #--------------------------------------------------------------------------------------------------------------------------------------------
                        # ---------------------------------------------------- 放字(單雙手開車合併) ------------------------------------------------------------------------------------
                        # # 用來儲存合併後的顯示文字和對應的最大值
                        # combined_text = "no_hand/single_hand_driving"
                        # max_display_value = None
                        # blank = 0
                        # # 遍歷所有 state 和 condition
                        # for state, condition in enumerate(result_list):
                        #     # 取得對應的 display_text，若 condition 為 False 則顯示 "0"
                        #     # current_display_value = 0 if not condition else out[state].item()
                        #     current_display_value = out[state].item()
                            
                        #     # 檢查當前的 cls_text[state] 是否為 "no_hand_driving" 或 "single_hand_driving"
                        #     if cls_text[state] in ["no_hand_driving", "single_hand_driving"]:
                        #         # 如果是其中之一，則比較當前 display_value 和最大值，保留較大的
                        #         blank = blank + 1
                        #         if max_display_value is None or current_display_value > max_display_value:
                        #             max_display_value = current_display_value
                        #         continue  # 跳過該狀態，避免重複繪製
                        #     else:
                        #         # 如果不是這兩者之一，則正常處理
                        #         display_text = "0" if not condition else "{:.2f}".format(current_display_value)
                        #         text = "{} : {}".format(cls_text[state], display_text)
                        #         color = cls_color[state]

                        #         # 在 frame 上绘制文本
                        #         cv2.putText(frame, text, (10, 30 + (state - blank)* 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # # 最後，如果 `max_display_value` 有值，則繪製合併後的文字
                        # if max_display_value is not None:
                        #     display_text = "{:.2f}".format(max_display_value)
                        #     text = "{} : {}".format(combined_text, display_text)
                        #     color = cls_color[state]  # 你可以選擇任何你認為合適的顏色

                        #     # 在 frame 上绘制合併後的文本
                        #     cv2.putText(frame, text, (10, 30 + (len(result_list) - blank) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        
                        
                        # ----------------------------------------------------- 放icon ----------------------------------------------------------------------------------
                        state_distract = False
                        state_phone = False
                        state_drowsy = False
                        state_nap = False
                        state_laughing = False
                        state_negative = False
                        state_no_seatbelt = False
                        state_seatbelt = False
                        state_single_hand = False
                        state_no_hand = False
                        state_singing = False
                        state_sneezing = False
                        state_eyescratching = False
                        state_talking = False
                        state_eating = False
                        state_smoking = False

                        
                        value_distract = 0
                        value_phone = 0
                        value_drowsy = 0
                        value_nap = 0
                        value_laughing = 0
                        value_negative = 0
                        value_no_seatbelt = 0
                        value_seatbelt = 0
                        value_single_hand = 0
                        value_no_hand = 0
                        value_singing = 0
                        value_sneezing = 0
                        value_eyescratching = 0
                        value_talking = 0
                        value_eating = 0
                        value_smoking = 0

                        # print("result_list = ",result_list)
                        for state, condition in enumerate(result_list):
                            # print("state = ",state)
                            # print("condition = ",condition)
                            if condition: 
                                # if cls_text[state] == 'no_hand_driving':
                                if state == 0 :
                                    state_no_hand = True
                                    value_no_hand = float("{:.1f}".format(out[state].item()))

                                # elif cls_text[state] == 'nap':
                                elif state == 1 :
                                    state_nap = True
                                    value_nap = float("{:.1f}".format(out[state].item()))

                                # elif cls_text[state] == 'seatbelt':
                                elif state == 2 :
                                    state_seatbelt = True
                                    value_seatbelt = float("{:.1f}".format(out[state].item()))

                                # elif cls_text[state] == 'Distract':
                                elif state == 3 :
                                    state_distract = True
                                    value_distract = float("{:.1f}".format(out[state].item()))
                                    
                                # elif cls_text[state] == 'Phone':
                                elif state == 4 :
                                    state_phone = True
                                    value_phone = float("{:.1f}".format(out[state].item()))
                                    
                                # elif cls_text[state] == 'Drowsy':
                                elif state == 5 :
                                    state_drowsy = True
                                    value_drowsy = float("{:.1f}".format(out[state].item()))
                                    
                                # elif cls_text[state] == 'laughing':
                                elif state == 6 :
                                    state_laughing = True
                                    value_laughing = float("{:.1f}".format(out[state].item()))

                                # elif cls_text[state] == 'single_hand_driving':
                                elif state == 7 :
                                    state_single_hand = True
                                    value_single_hand = float("{:.1f}".format(out[state].item()))
                                    
                                # elif cls_text[state] == 'negative':
                                elif state == 8 :
                                    state_negative = True
                                    value_negative = float("{:.1f}".format(out[state].item()))

                                # elif cls_text[state] == 'no_seatbelt':
                                # elif state == 9 :
                                #     state_no_seatbelt = True
                                #     value_no_seatbelt = float("{:.1f}".format(out[state].item()))
                                
                                elif state == 9 :
                                    state_singing = True
                                    value_singing = float("{:.1f}".format(out[state].item()))

                                elif state == 10 :
                                    state_sneezing = True
                                    value_sneezing = float("{:.1f}".format(out[state].item()))

                                elif state == 11 :
                                    state_eyescratching = True
                                    value_eyescratching = float("{:.1f}".format(out[state].item()))

                                elif state == 12 :
                                    state_talking = True
                                    value_talking = float("{:.1f}".format(out[state].item()))

                                elif state == 13 :
                                    state_eating = True
                                    value_eating = float("{:.1f}".format(out[state].item()))

                                elif state == 14 :
                                    state_smoking = True
                                    value_smoking = float("{:.1f}".format(out[state].item()))

                                    
                                
                            
                            if(state == class_num):
                                # print("############ buffer push ################")
                                # print("value_distract:",value_distract)
                                # print("value_phone:",value_phone)
                                # print("value_drowsy:",value_drowsy)
                                # print("value_laughing:",value_laughing)
                                # print("value_seatbelt:",value_seatbelt)
                                # print("value_nap:",value_nap)
                                # print("value_single_hand:",value_single_hand)
                                # print("value_no_hand:",value_no_hand)
                                # print("value_singing:",value_singing)
                                # print("value_sneezing:",value_sneezing)
                                # print("value_eyescratching:",value_eyescratching)
                                # print("value_talking:",value_talking)
                                # print("value_eating:",value_eating)
                                # print("value_smoking:",value_smoking)
                                
                                # print("state_distract:",state_distract)
                                # print("state_phone:",state_phone)
                                # print("state_drowsy:",state_drowsy)
                                # print("state_laughing:",state_laughing)
                                # print("state_seatbelt:",state_seatbelt)
                                # print("state_nap:",state_nap)
                                # print("state_single_hand:",state_single_hand)
                                # print("state_no_hand:",state_no_hand)
                                # print("state_singing:",value_singing)
                                # print("state_sneezing:",value_sneezing)
                                # print("state_eyescratching:",value_eyescratching)
                                # print("state_talking:",value_talking)
                                # print("state_eating:",value_eating)
                                # print("state_smoking:",value_smoking)
                                
                                # print("buffer_distract:",buffer_distract.count(1))
                                # print("buffer_phone:",buffer_phone.count(1))
                                # print("buffer_drowsy:",buffer_drowsy.count(1))
                                # print("buffer_nap:",buffer_nap.count(1))
                                # print("buffer_laughing:",buffer_laughing.count(1))
                                # print("buffer_seatbelt:",buffer_seatbelt.count(1))
                                # print("buffer_steer_wheel:",buffer_steer_wheel.count(1))
                                # print("buffer_singing:",buffer_singing.count(1))
                                # print("buffer_sneezing:",buffer_sneezing.count(1))
                                # print("buffer_eyescratching:",buffer_eyescratching.count(1))
                                # print("buffer_talking:",buffer_talking.count(1))
                                # print("buffer_eating:",buffer_eating.count(1))
                                # print("buffer_smoking:",buffer_smoking.count(1))

                                
                                buffer_distract.pop(0)
                                buffer_phone.pop(0)
                                buffer_drowsy.pop(0)
                                buffer_nap.pop(0)
                                buffer_laughing.pop(0)
                                buffer_seatbelt.pop(0)
                                buffer_steer_wheel.pop(0)  
                                buffer_singing.pop(0) 
                                buffer_sneezing.pop(0) 
                                buffer_eyescratching.pop(0) 
                                buffer_talking.pop(0) 
                                buffer_eating.pop(0) 
                                buffer_smoking.pop(0) 
                                buffer_negative.pop(0)

                                ## multilabel
                                if(value_distract > -2):
                                # if(state_distract == True and value_distract > 0.4):
                                    buffer_distract.append(1)
                                else:
                                    buffer_distract.append(0)
                                
                                if(value_phone > 0.6):
                                # if(state_phone == True and value_phone > 0.2):
                                    buffer_phone.append(1)
                                else:
                                    buffer_phone.append(0)
                                    
                                if((value_drowsy > -1 and value_laughing < -2) or value_drowsy > 1.5):
                                # if(state_drowsy == True and value_drowsy > 0.4):
                                    buffer_drowsy.append(1)
                                else:
                                    buffer_drowsy.append(0)

                                if( value_nap > 1 and value_distract < -2 and value_sneezing < -4):
                                # if(state_nap == True and value_nap > 0.3):
                                    buffer_nap.append(1)
                                else:
                                    buffer_nap.append(0)
                                    
                                if(value_laughing > -2.5 and value_drowsy < -1):   #laughing == -3 (11/4 15:34) #laughing == -2.5 (11/4 15:35)
                                # if(state_laughing == True and value_laughing > 0.2):
                                    buffer_laughing.append(1)
                                else:
                                    buffer_laughing.append(0)
                                    
                                if(value_seatbelt < -1):
                                    buffer_seatbelt.append(1)
                                else:
                                    buffer_seatbelt.append(0)
                                
                                if value_negative > 0:
                                    buffer_negative.append(1)
                                else:
                                    buffer_negative.append(0)


                                if value_negative < 1 and value_negative > -5:
                                    if((value_single_hand > 4 or value_no_hand > 0) and (buffer_negative.count(1)<buffer_negative.count(0))):
                                        buffer_steer_wheel.append(1)
                                    else:
                                        buffer_steer_wheel.append(0)

                                elif value_negative < -5:
                                    if((value_single_hand > 0 or value_no_hand > 0) and (buffer_negative.count(1)<buffer_negative.count(0))):
                                        buffer_steer_wheel.append(1)
                                    else:
                                        buffer_steer_wheel.append(0)
                                else:
                                    buffer_steer_wheel.append(0)


                                    
                                

                                if(value_singing > 0):
                                    buffer_singing.append(1)
                                else:
                                    buffer_singing.append(0)

                                if(value_sneezing > -4 and value_nap < 1):
                                # if(state_sneezing == True and value_sneezing > 0.2):
                                    buffer_sneezing.append(1)
                                else:
                                    buffer_sneezing.append(0)

                                if(value_eyescratching > -4 and value_phone < 0.6 and value_eating < -2 and value_smoking < -3):
                                # if(state_eyescratching == True and value_eyescratching > 0.1):
                                    buffer_eyescratching.append(1)
                                else:
                                    buffer_eyescratching.append(0)

                                if(value_talking > -3.5 and value_laughing < -3):
                                # if(state_talking == True and value_talking > 0.4):
                                    buffer_talking.append(1)
                                else:
                                    buffer_talking.append(0)

                                if(value_eating > -2 and value_smoking < -3  and value_eyescratching < -4):
                                # if(state_eating == True and value_eating > 0.2):
                                    buffer_eating.append(1)
                                else:
                                    buffer_eating.append(0)

                                if(value_smoking > -3 and value_eating < -4 and value_eyescratching < -4 and value_phone < 0.6 ):
                                # if(state_smoking == True and value_smoking > 0.2 and value_smoking > value_eating and value_smoking > value_eyescratching):
                                    buffer_smoking.append(1)
                                else:
                                    buffer_smoking.append(0)




                                ## long_distraction:continuous 3 seconds for distraction, nap, phone
                                ## short_distraction: cumulative 10 seconds for distraction, nap, phone within a 30 second time(reset if attention buffer full)
                                if(play_icon_distract.count(1) > 0) or (play_icon_nap.count(1) > 0) or (play_icon_phone.count(1) > 0):
                                    buffer_long_distraction.append(1)
                                    buffer_short_distraction.append(1)
                                    buffer_attention.append(0)
                                    
                                    buffer_long_distraction.pop(0)
                                    buffer_short_distraction.pop(0)
                                    buffer_attention.pop(0)                                    
                                else:
                                    for i in range(len(buffer_long_distraction)):
                                        buffer_long_distraction[i] = 0
                                    buffer_short_distraction.append(0)
                                    buffer_attention.append(1)
                                    buffer_short_distraction.pop(0)
                                    buffer_attention.pop(0)

                                if(buffer_attention.count(0)==0):
                                    for i in range(len(buffer_short_distraction)):
                                        buffer_short_distraction[i] = 0

                                ## For KSS level
                                buffer_drowsy_KSS_flag.pop(0)
                                buffer_drowsy_KSS.pop(0)
                                buffer_nap_KSS_flag.pop(0)
                                buffer_nap_KSS.pop(0)

                                if(play_icon_drowsy.count(1)==len(play_icon_drowsy)):
                                    buffer_drowsy_KSS_flag.append(1)
                                else:
                                    buffer_drowsy_KSS_flag.append(0)

                                if(play_icon_nap.count(1)==len(play_icon_nap)):
                                    buffer_nap_KSS_flag.append(1)
                                else:
                                    buffer_nap_KSS_flag.append(0)

                                if(buffer_drowsy_KSS_flag == [0,1]):
                                    buffer_drowsy_KSS.append(1)
                                else:    
                                    buffer_drowsy_KSS.append(0)

                                if(buffer_nap_KSS_flag == [0,1]):
                                    buffer_nap_KSS.append(1)
                                else:    
                                    buffer_nap_KSS.append(0)

                                    

                                #     text = "{}".format("clean")
                                #     cv2.putText(frame, text, (70, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                # if(buffer_attention.count(0)==0):
                                #     for i in range(len(buffer_short_distraction)):
                                #         buffer_short_distraction[i] = 0

                                # if(play_icon_distract.count(1) > 0) or (play_icon_nap.count(1) > 0) or (play_icon_phone.count(1) > 0):
                                #     if(buffer_long_distraction < 3*fps_real):
                                #         buffer_long_distraction += 1
                                # else:
                                #     buffer_long_distraction = 0
                                #     # text = "{}".format("clean")
                                #     # cv2.putText(frame, text, (70, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    

                                # if(play_icon_distract.count(1) > 0) or (play_icon_nap.count(1) > 0) or (play_icon_phone.count(1) > 0):
                                #     if(buffer_short_distraction < 30*fps_real):
                                #         buffer_short_distraction += 1
                                # # else:
                                #     # if(buffer_short_distraction > 0):
                                #     #     buffer_short_distraction -= 1

                                # if(play_icon_distract.count(1) > 0) or (play_icon_nap.count(1) > 0) or (play_icon_phone.count(1) > 0):
                                #     if(buffer_attention > 0):
                                #         buffer_attention -= 1
                                # else:
                                #     if(buffer_attention < 2*fps_real):
                                #         buffer_attention += 1

                                # if(buffer_attention == 2*fps_real):
                                #     buffer_short_distraction = 0

                                # distraction_time = "{:.2f}".format(play_icon_distract.count(1)/fps_real)
                                # text = "{} : {}".format("distraction : ", distraction_time)
                                # color = (255, 255, 255)
                                # cv2.putText(frame, text, (100, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                # nap_time = "{:.2f}".format(play_icon_nap.count(1)/fps_real)
                                # text = "{} : {}".format("nap : ", nap_time)
                                # color = (255, 255, 255)
                                # cv2.putText(frame, text, (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                # phone_time = "{:.2f}".format(play_icon_phone.count(1)/fps_real)
                                # text = "{} : {}".format("phone : ", phone_time)
                                # color = (255, 255, 255)
                                # cv2.putText(frame, text, (100, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                long_distraction_time = "{:.2f}".format(buffer_long_distraction.count(1)/fps_real)
                                # long_distraction_time = "{:.2f}".format(buffer_long_distraction/fps_real)
                                text = "{} : {}".format("Long_distraction ", long_distraction_time)
                                if(buffer_long_distraction.count(1) == 3*fps_real):
                                    long = True
                                    color = (0, 0, 255) ##BGR
                                else:
                                    long = False
                                    color = (255, 255, 255)
                                cv2.putText(frame, text, (100, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                short_distraction_time = "{:.2f}".format(buffer_short_distraction.count(1)/fps_real)
                                # short_distraction_time = "{:.2f}".format(buffer_short_distraction/fps_real)
                                text = "{} : {}".format("Short_distraction (cumulative) ", short_distraction_time)
                                if(buffer_short_distraction.count(1) >= 10*fps_real):
                                    color = (0, 0, 255)
                                    short = True
                                else:
                                    color = (255, 255, 255)
                                    short = False
                                cv2.putText(frame, text, (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                attention_time = "{:.2f}".format(buffer_attention.count(1)/fps_real)
                                # attention_time = "{:.2f}".format(buffer_attention/fps_real)
                                text = "{} : {}".format("Attention Buffer ", attention_time)
                                if(buffer_attention.count(1) == 2*fps_real):
                                    color = (0, 255, 0)
                                else:
                                    color = (255, 255, 255)
                                cv2.putText(frame, text, (100, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                drowsy_num = buffer_drowsy_KSS.count(1)
                                text = "{} : {}".format("Drowsy num for 1 min ", drowsy_num)
                                if(drowsy_num == 0):
                                    color = (255, 255, 255)
                                    KSS_drowsy = 5
                                elif(drowsy_num>0 and drowsy_num <= 3):
                                    color = (0, 140, 255) #orange
                                    # color = (255, 255, 255)
                                    KSS_drowsy = 6
                                else:
                                    color = (0,0,255)  #red
                                    # color = (255, 255, 255)
                                    KSS_drowsy = 7
                                cv2.putText(frame, text, (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                nap_num = buffer_nap_KSS.count(1)
                                text = "{} : {}".format("Nap num for 1 min ", nap_num)
                                if(nap_num == 0):
                                    color = (255, 255, 255)
                                    KSS_nap = 5
                                elif(nap_num <= 2):
                                    color = (0, 140, 255)
                                    KSS_nap = 8
                                else:
                                    color = (0,0,255)
                                    KSS_nap = 9
                                cv2.putText(frame, text, (100, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                KSS_level = max(KSS_nap,KSS_drowsy)            #KSS完整版
                                if(KSS_level == 5):
                                    color = (255, 255, 255)
                                elif(KSS_level == 6 or KSS_level ==7):
                                    color = (0, 140, 255)
                                else:    
                                    color = (0,0,255)

                                # KSS_level = KSS_drowsy                           #KSS只有drowsy                          
                                # color = (255, 255, 255)
                                

                                text = "{} : {}".format("KSS level : ", KSS_level)
                                cv2.putText(frame, text, (100, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                play_icon_distract.pop(0) 
                                play_icon_distract.append(0) 
                                play_icon_phone.pop(0)  
                                play_icon_phone.append(0)  
                                play_icon_drowsy.pop(0)
                                play_icon_drowsy.append(0)
                                play_icon_nap.pop(0)
                                play_icon_nap.append(0)
                                play_icon_laughing.pop(0)
                                play_icon_laughing.append(0)
                                play_icon_seatbelt.pop(0)
                                play_icon_seatbelt.append(0)
                                play_icon_steer_wheel.pop(0) 
                                play_icon_steer_wheel.append(0) 
                                play_icon_singing.pop(0) 
                                play_icon_singing.append(0) 
                                play_icon_sneezing.pop(0) 
                                play_icon_sneezing.append(0) 
                                play_icon_eyescratching.pop(0) 
                                play_icon_eyescratching.append(0) 
                                play_icon_talking.pop(0) 
                                play_icon_talking.append(0) 
                                play_icon_eating.pop(0) 
                                play_icon_eating.append(0) 
                                play_icon_smoking.pop(0) 
                                play_icon_smoking.append(0) 

                                #icon放右邊一行
                                if(buffer_distract.count(1)>buffer_distract.count(0)):
                                    for i in range(len(play_icon_distract)):
                                        play_icon_distract[i] = 1
                                
                                if(play_icon_distract.count(1) > 0):
                                    icon_sequence = 0 #調整位置用
                                    print('icon = icon_distract')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_distract_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_1)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_distract_resized.shape[0] <= frame.shape[0] and icon_x_1 + icon_distract_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_distract_resized.shape[0], icon_x_1:icon_x_1+icon_distract_resized.shape[1]] = icon_distract_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')
                                        
                                if(buffer_phone.count(1)>buffer_phone.count(0)):
                                    for i in range(len(play_icon_phone)):
                                        play_icon_phone[i] = 1
                                
                                if(play_icon_phone.count(1) > 0):
                                    icon_sequence = 1 #調整位置用
                                    print('icon = icon_phone')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_phone_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_1)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_phone_resized.shape[0] <= frame.shape[0] and icon_x_1 + icon_phone_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_phone_resized.shape[0], icon_x_1:icon_x_1+icon_phone_resized.shape[1]] = icon_phone_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')
                                
                                if(buffer_nap.count(1)>buffer_nap.count(0)):
                                    for i in range(len(play_icon_nap)):
                                        play_icon_nap[i] = 1

                                if((buffer_drowsy.count(1)>buffer_drowsy.count(0))):
                                    for i in range(len(play_icon_drowsy)):
                                        play_icon_drowsy[i] = 1
                                
                                
                                if(play_icon_drowsy.count(1) > 0 or play_icon_nap.count(1)):
                                    icon_sequence = 2 #調整位置用
                                    print('icon = icon_drowsy')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_drowsy_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_1)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_drowsy_resized.shape[0] <= frame.shape[0] and icon_x_1 + icon_drowsy_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_drowsy_resized.shape[0], icon_x_1:icon_x_1+icon_drowsy_resized.shape[1]] = icon_drowsy_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')
                                        
                                if(buffer_seatbelt.count(1)>buffer_seatbelt.count(0)):
                                    for i in range(len(play_icon_seatbelt)):
                                        play_icon_seatbelt[i] = 1

                                if(play_icon_seatbelt.count(1) > 0):
                                    icon_sequence = 3 #調整位置用
                                    print('icon = icon_seatbelt')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_seatbelt_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_1)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_seatbelt_resized.shape[0] <= frame.shape[0] and icon_x_1 + icon_seatbelt_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_seatbelt_resized.shape[0], icon_x_1:icon_x_1+icon_seatbelt_resized.shape[1]] = icon_seatbelt_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')
                                        
                                if(buffer_laughing.count(1)>buffer_laughing.count(0)):
                                    for i in range(len(play_icon_laughing)):
                                        play_icon_laughing[i] = 1

                                if(play_icon_laughing.count(1) > 0):
                                    icon_sequence = 4 #調整位置用
                                    print('icon = icon_laughing')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_laughing_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_1)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_laughing_resized.shape[0] <= frame.shape[0] and icon_x_1 + icon_laughing_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_laughing_resized.shape[0], icon_x_1:icon_x_1+icon_laughing_resized.shape[1]] = icon_laughing_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')
                                        
                                if(buffer_steer_wheel.count(1)>buffer_steer_wheel.count(0)):
                                    for i in range(len(play_icon_steer_wheel)):
                                        play_icon_steer_wheel[i] = 1
                                
                                if(play_icon_steer_wheel.count(1) > 0):
                                    icon_sequence = 5 #調整位置用
                                    print('icon = icon_steer_wheel')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_steer_wheel_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_1)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_steer_wheel_resized.shape[0] <= frame.shape[0] and icon_x_1 + icon_steer_wheel_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_steer_wheel_resized.shape[0], icon_x_1:icon_x_1+icon_steer_wheel_resized.shape[1]] = icon_steer_wheel_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')
                        
                                #icon放左邊一行
                                # if(buffer_singing.count(1)>buffer_singing.count(0)):
                                #     for i in range(len(play_icon_singing)):
                                #         play_icon_singing[i] = 1
                                
                                # if(play_icon_singing.count(1) > 0):
                                #     icon_sequence = 0 #調整位置用
                                #     print('icon = icon_singing')
                                #     print ("frame shape = ",frame.shape)
                                #     icon_y = icon_y_start + icon_sequence * (icon_singing_resized.shape[0] + icon_spacing)
                                #     print(cls_text)
                                #     print("icon x:",icon_x_2)    
                                #     print("icon y:",icon_y)    
                                #     # 確保圖標放置的區域不超出 frame 的範圍
                                #     if icon_y + icon_steer_wheel_resized.shape[0] <= frame.shape[0] and icon_x_2 + icon_singing_resized.shape[1] <= frame.shape[1]:
                                #         # 将图标放置在 frame 上
                                #         frame[icon_y:icon_y+icon_singing_resized.shape[0], icon_x_2:icon_x_2+icon_singing_resized.shape[1]] = icon_singing_resized
                                #         print('put icon success')
                                #     else:
                                #         print('icon is out of bound')

                                if(buffer_sneezing.count(1)>buffer_sneezing.count(0)):
                                    for i in range(len(play_icon_sneezing)):
                                        play_icon_sneezing[i] = 1
                                
                                if(play_icon_sneezing.count(1) > 0):
                                    icon_sequence = 1 #調整位置用
                                    print('icon = icon_sneezing')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_sneezing_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_2)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_steer_wheel_resized.shape[0] <= frame.shape[0] and icon_x_2 + icon_sneezing_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_sneezing_resized.shape[0], icon_x_2:icon_x_2+icon_sneezing_resized.shape[1]] = icon_sneezing_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')
                                
                                if(buffer_eyescratching.count(1)>buffer_eyescratching.count(0)):
                                    for i in range(len(play_icon_eyescratching)):
                                        play_icon_eyescratching[i] = 1
                                
                                if(play_icon_eyescratching.count(1) > 0):
                                    icon_sequence = 2 #調整位置用
                                    print('icon = icon_eyescratching')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_eyescratching_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_2)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_steer_wheel_resized.shape[0] <= frame.shape[0] and icon_x_2 + icon_eyescratching_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_eyescratching_resized.shape[0], icon_x_2:icon_x_2+icon_eyescratching_resized.shape[1]] = icon_eyescratching_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')

                                if(buffer_talking.count(1)>buffer_talking.count(0) or buffer_singing.count(1)>buffer_singing.count(0)):
                                    for i in range(len(play_icon_talking)):
                                        play_icon_talking[i] = 1
                                
                                if(play_icon_talking.count(1) > 0):
                                    icon_sequence = 3 #調整位置用
                                    print('icon = icon_talking')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_talking_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_2)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_steer_wheel_resized.shape[0] <= frame.shape[0] and icon_x_2 + icon_talking_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_talking_resized.shape[0], icon_x_2:icon_x_2+icon_talking_resized.shape[1]] = icon_talking_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')

                                if(buffer_eating.count(1)>buffer_eating.count(0)):
                                    for i in range(len(play_icon_eating)):
                                        play_icon_eating[i] = 1
                                
                                if(play_icon_eating.count(1) > 0):
                                    icon_sequence = 4 #調整位置用
                                    print('icon = icon_eating')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_eating_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_2)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_steer_wheel_resized.shape[0] <= frame.shape[0] and icon_x_2 + icon_eating_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_eating_resized.shape[0], icon_x_2:icon_x_2+icon_eating_resized.shape[1]] = icon_eating_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')

                                if(buffer_smoking.count(1)>buffer_smoking.count(0)):
                                    for i in range(len(play_icon_smoking)):
                                        play_icon_smoking[i] = 1
                                
                                if(play_icon_smoking.count(1) > 0):
                                    icon_sequence = 5 #調整位置用
                                    print('icon = icon_smoking')
                                    print ("frame shape = ",frame.shape)
                                    icon_y = icon_y_start + icon_sequence * (icon_smoking_resized.shape[0] + icon_spacing)
                                    print(cls_text)
                                    print("icon x:",icon_x_2)    
                                    print("icon y:",icon_y)    
                                    # 確保圖標放置的區域不超出 frame 的範圍
                                    if icon_y + icon_steer_wheel_resized.shape[0] <= frame.shape[0] and icon_x_2 + icon_smoking_resized.shape[1] <= frame.shape[1]:
                                        # 将图标放置在 frame 上
                                        frame[icon_y:icon_y+icon_smoking_resized.shape[0], icon_x_2:icon_x_2+icon_smoking_resized.shape[1]] = icon_smoking_resized
                                        print('put icon success')
                                    else:
                                        print('icon is out of bound')

                                
                        # ---------------------------------------------------------------------------------------------------------------------------------------
                                    

                    # cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE) 
                    # cv2.imshow('frame', frame)
                    cv2.imwrite(out_path + result+'/frame'+str(frame_numbers).zfill(2)+'.jpg', frame)
                    if counter == number_of_skip_frame:
                        counter = 0
                    start_time = time.time()

                    #####################################################################################
                    #if inference_flag == True:
                        #print('after model: {}'.format(time.time()-after_model_time_start))
                    #print('total time: {}'.format(time.time()-read_img_time_start))
                    ###########################################################################


                time.sleep(1 / fps)
                # time.sleep(2/fps) # modifiy the number to show image more slowly in order to observe the output
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        t2= time.time()
        print('time usage:')
        print(t2-t1)
        # print('FPS:')
        # print(60/(t2-t1))

        def sort_key(s):
                        return int(s.split('frame')[1].split('.jpg')[0])
        
        folder_path = os.path.join(out_path+result)
        output_directory = outputpath

        # 檢查輸出目錄是否存在
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # 讀取所有 frame 開頭的圖片並排序
        images = [img for img in os.listdir(folder_path) if img.startswith("frame")]
        images.sort(key=sort_key)

        # 檢查是否有圖片
        if len(images) == 0:
            print(f"No images found in {folder_path}!")
        else:
            print(images)

            # 讀取第一張圖片取得尺寸
            frame = cv2.imread(os.path.join(folder_path, images[0]))
            h, w, layers = frame.shape
            size = (w, h)

            # 設定輸出影片路徑
            output_video_path = os.path.join(output_directory, f"{os.path.basename(result)}.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

            # 寫入所有圖片
            for image in images:
                img_path = os.path.join(folder_path, image)
                img = cv2.imread(img_path)
                out.write(img)
                print(f"Writing frame {image}")

            # 釋放資源
            out.release()
            print(f"Video {output_video_path} created successfully!")





    # dummy_input = torch.randn(1, 24, 256, 224, device="cuda")

    # input_names = [ 'input' ]
    # output_names = ['input_for_heatmap','base_out', 'output']


    # torch.onnx.export(model.module, dummy_input, "mobilenetv2_classification_tsm.onnx", verbose=False, input_names=input_names, output_names=output_names)


    # # summary(model.module, (24, 256, 256,))


    # flops, params = profile(model.module, inputs=(dummy_input, ))
    # flops, params = clever_format([flops, params], "%.3f")
    # print('flops:', flops)
    # print('params:', params)


if __name__ == "__main__":
    main()

