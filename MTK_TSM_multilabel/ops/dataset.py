# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cv2


class VideoRecord(object):
    def __init__(self, row):
        # print(row)
        row[2] = ''.join(row[2:]).replace(', ', ',')  # 合併並且確保每個元素之間只有一個逗號
        row = row[:3]  # 去掉多餘的元素
        self._data = row
        # print(self._data)
    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        # print(self._data[2])
        return (self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', label_tmpl = 'img_{:05d}',  transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False, two_stream = False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.label_tmpl = label_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        self.two_stream = two_stream
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()
        
    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:

                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'GRAY':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('L')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('L')]
        elif self.modality == 'YUV' :
            try:
                image = cv2.imread(os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                grey, img_mvx , img_mvy  = cv2.split(image)

                img_128 = np.empty([480, 640], dtype=np.uint8)
                img_128.fill(128)

                
                image_merged_grey = cv2.merge([grey, grey, grey])
                image_merged_motion = cv2.merge([img_128, img_mvx, img_mvy])
                image_merged = cv2.merge([grey, img_mvx, img_mvy])
                image_grey = Image.fromarray(image_merged_grey)
                image_motion = Image.fromarray(image_merged_motion)
                image_merged = Image.fromarray(image_merged)
                if self.two_stream == True:
                    return [image_grey], [image_motion]
                else:
                    return [image_merged]
                
                
                
                
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
            
        self.video_list = [VideoRecord(item) for item in tmp]
        
        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        
        record = self.video_list[index]
        # check this is a legit video folder

        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        # print('full_path:', full_path)
        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get_classification(record, segment_indices)

    def get_classification(self, record, indices):
        
        #load images
        images = list()
        images_motion = list()


        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                if self.modality == 'YUV' and self.two_stream == True:
                    seg_imgs_grey, seg_imgs_motion = self._load_image(record.path, p)
                    images.extend(seg_imgs_grey)
                    images_motion.extend(seg_imgs_motion)               
                else:
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        # print('==================================================================================')
        # print(self.transform)
        process_data = self.transform(images)
        if self.modality == 'YUV' and self.two_stream == True:
            process_data_motion = self.transform(images_motion)

        if self.modality == 'YUV' and self.two_stream == True:
            return process_data, process_data_motion, record.label # for classfication 
        else:
            return process_data, record.label # for classfication 
        
    def __len__(self):
        return len(self.video_list)


# class TSN_Heatmap_DataSet(data.Dataset):
#     def __init__(self, root_path, list_file,
#                  num_segments=3, new_length=1, modality='RGB',
#                  image_tmpl='img_{:05d}.jpg', label_tmpl = 'img_{:05d}',  transform=None,
#                  random_shift=True, test_mode=False,
#                  remove_missing=False, dense_sample=False, twice_sample=False):

#         self.root_path = root_path
#         self.list_file = list_file
#         self.num_segments = num_segments
#         self.new_length = new_length
#         self.modality = modality
#         self.image_tmpl = image_tmpl
#         self.label_tmpl = label_tmpl
#         self.transform = transform
#         self.random_shift = random_shift
#         self.test_mode = test_mode
#         self.remove_missing = remove_missing
#         self.dense_sample = dense_sample  # using dense sample as I3D
#         self.twice_sample = twice_sample  # twice sample for more validation
#         self.record_path = ''
#         if self.dense_sample:
#             print('=> Using dense sample for the dataset...')
#         if self.twice_sample:
#             print('=> Using twice sample for the dataset...')

#         if self.modality == 'RGBDiff':
#             self.new_length += 1  # Diff needs one more image to calculate diff

#         self._parse_list()

#     ############################################################
#     def _load_heatmap_label(self, idx):
#         if self.modality == 'RGB' or self.modality == 'RGBDiff':
#             try:
#                 behavior_cnt = 4  # a, b, c, d

#                 #label_prefix_no_txt = /home/s310511029/dataset/behaviour_dataset_img/emergencybrake/v_emergencybrake_g29_c05/image_00004

#                 label_prefix_no_txt = os.path.join(self.record_path, self.label_tmpl.format(idx))
#                 fn_a = label_prefix_no_txt + '_a.txt'
#                 fn_b = label_prefix_no_txt + '_b.txt'
#                 fn_c = label_prefix_no_txt + '_c.txt'
#                 fn_d = label_prefix_no_txt + '_d.txt'
                
#                 arg = [fn_a, fn_b, fn_c, fn_d]
#                 with ThreadPoolExecutor(max_workers=behavior_cnt) as executor:
#                     result = executor.map(self._load_label_file, arg)
#                 res = [r for r in result]  # [a, b, c, d]
                
#                 return res
                      
#             except Exception:
#                 label_prefix_no_txt = os.path.join(self.record_path, self.label_tmpl.format(idx))
#                 print('error loading label:', label_prefix_no_txt)
#         else:
#             raise NotImplementedError("other modality is not yet implemented.")

#     def _load_label_file(self, filename):
#         out_map = np.loadtxt(filename)
#         return out_map


#     ##################################################################
#     def _load_image(self, directory, idx):
#         if self.modality == 'RGB' or self.modality == 'RGBDiff':
#             try:
                
#                 return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
#             except Exception:
#                 print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
#                 return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
#         elif self.modality == 'YUV' :
#             try:
#                 return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('YCbCr')]
#             except Exception:
#                 print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
#                 return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('YCbCr')]
#         elif self.modality == 'Flow':
#             if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
#                 x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
#                     'L')
#                 y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
#                     'L')
#             elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
#                 x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
#                                                 format(int(directory), 'x', idx))).convert('L')
#                 y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
#                                                 format(int(directory), 'y', idx))).convert('L')
#             else:
#                 try:
#                     # idx_skip = 1 + (idx-1)*5
#                     flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
#                         'RGB')
#                 except Exception:
#                     print('error loading flow file:',
#                           os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
#                     flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
#                 # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
#                 flow_x, flow_y, _ = flow.split()
#                 x_img = flow_x.convert('L')
#                 y_img = flow_y.convert('L')

#             return [x_img, y_img]

#     def _parse_list(self):
#         # check the frame number is large >3:
        
#         tmp = [x.strip().split(' ') for x in open(self.list_file)]
#         if not self.test_mode or self.remove_missing:
#             tmp = [item for item in tmp if int(item[1]) >= 3]
            
#         self.video_list = [VideoRecord(item) for item in tmp]

#         if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
#             for v in self.video_list:
#                 v._data[1] = int(v._data[1]) / 2
#         print('video number:%d' % (len(self.video_list)))

#     def _sample_indices(self, record):
#         """

#         :param record: VideoRecord
#         :return: list
#         """
#         if self.dense_sample:  # i3d dense sample
#             sample_pos = max(1, 1 + record.num_frames - 64)
#             t_stride = 64 // self.num_segments
#             start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
#             offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
#             return np.array(offsets) + 1
#         else:  # normal sample
#             average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            
#             if average_duration > 0:
#                 offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
#                                                                                                   size=self.num_segments)
#             elif record.num_frames > self.num_segments:
#                 offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
#             else:
#                 offsets = np.zeros((self.num_segments,))
#             return offsets + 1

#     def _get_val_indices(self, record):
#         if self.dense_sample:  # i3d dense sample
#             sample_pos = max(1, 1 + record.num_frames - 64)
#             t_stride = 64 // self.num_segments
#             start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
#             offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
#             return np.array(offsets) + 1
#         else:
#             if record.num_frames > self.num_segments + self.new_length - 1:
#                 tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
#                 offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
#             else:
#                 offsets = np.zeros((self.num_segments,))
#             return offsets + 1

#     def _get_test_indices(self, record):
#         if self.dense_sample:
#             sample_pos = max(1, 1 + record.num_frames - 64)
#             t_stride = 64 // self.num_segments
#             start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
#             offsets = []
#             for start_idx in start_list.tolist():
#                 offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
#             return np.array(offsets) + 1
#         elif self.twice_sample:
#             tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

#             offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
#                                [int(tick * x) for x in range(self.num_segments)])

#             return offsets + 1
#         else:
#             tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
#             offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
#             return offsets + 1

#     def __getitem__(self, index):
        
#         record = self.video_list[index]
#         # check this is a legit video folder

#         if self.image_tmpl == 'flow_{}_{:05d}.jpg':
#             file_name = self.image_tmpl.format('x', 1)
#             full_path = os.path.join(self.root_path, record.path, file_name)
#         elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
#             file_name = self.image_tmpl.format(int(record.path), 'x', 1)
#             full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
#         else:
#             file_name = self.image_tmpl.format(1)
#             full_path = os.path.join(self.root_path, record.path, file_name)
#         # print('full_path:', full_path)
#         while not os.path.exists(full_path):
#             print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
#             index = np.random.randint(len(self.video_list))
#             record = self.video_list[index]
#             if self.image_tmpl == 'flow_{}_{:05d}.jpg':
#                 file_name = self.image_tmpl.format('x', 1)
#                 full_path = os.path.join(self.root_path, record.path, file_name)
#             elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
#                 file_name = self.image_tmpl.format(int(record.path), 'x', 1)
#                 full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
#             else:
#                 file_name = self.image_tmpl.format(1)
#                 full_path = os.path.join(self.root_path, record.path, file_name)

#         if not self.test_mode:
#             segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
#         else:
#             segment_indices = self._get_test_indices(record)
#         return self.get_heatmap(record, segment_indices)

#     def get_classification(self, record, indices):
        
#         #load images
#         images = list()

#         for seg_ind in indices:
#             p = int(seg_ind)
#             for i in range(self.new_length):
#                 seg_imgs = self._load_image(record.path, p)
#                 images.extend(seg_imgs)
#                 if p < record.num_frames:
#                     p += 1

#         # print('==================================================================================')
#         # print(self.transform)
#         process_data = self.transform(images)

#         return process_data, record.label # for classfication 
        

#     def get_heatmap(self, record, indices):
        
#         #load images
#         images = list()
#         train_txts_a, train_txts_b, train_txts_c, train_txts_d = [], [], [], []

#         for seg_ind in indices:
#             p = int(seg_ind)
#             for i in range(self.new_length):
#                 seg_imgs = self._load_image(record.path, p)
#                 images.extend(seg_imgs)
#                 if p < record.num_frames:
#                     p += 1

#         #load heatmap labels
#         """ START read label .txt """
#         with ThreadPoolExecutor(max_workers=len(indices)) as executor:
#             self.record_path = record.path
#             print(indices)
#             result = executor.map(self._load_heatmap_label, list(indices))
        
#         res = [r for r in result]# (8,4,256)
#         train_txts_a, train_txts_b, train_txts_c, train_txts_d = list(zip(*res))# (8,256),tuple
        
#         """ END read label .txt """    


#         # print('==================================================================================')
#         # print(self.transform)
#         process_data = self.transform(images)

#         train_txts_a = np.array(train_txts_a, dtype=np.float32)
#         train_txts_b = np.array(train_txts_b, dtype=np.float32)
#         train_txts_c = np.array(train_txts_c, dtype=np.float32)
#         train_txts_d = np.array(train_txts_d, dtype=np.float32)

#         # return process_data, record.label # for classfication 
#         return process_data, train_txts_a, train_txts_b, train_txts_c, train_txts_d # for heatmap

#     def __len__(self):
#         return len(self.video_list)
