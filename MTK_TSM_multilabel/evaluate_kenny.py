import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter
from torch.nn.utils import clip_grad_norm_
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
from collections import defaultdict
def analyze_fp_cross_class(y_true, y_pred, class_names):
    """
    分析 false positive 對應的交叉類別：
    例如：某樣本實際只有類別 A，模型卻預測 A + B → B 就是 A 的 FP。
    """
    num_classes = y_true.shape[1]
    fp_confusion = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        true_classes = np.where(y_true[i] == 1)[0]
        pred_classes = np.where(y_pred[i] == 1)[0]

        # 每一個被預測但不在 GT 中的類別，都是 FP
        false_positives = set(pred_classes) - set(true_classes)

        for fp_cls in false_positives:
            for gt_cls in true_classes:
                # 這筆樣本真實是 gt_cls，但被誤預測了 fp_cls
                fp_confusion[gt_cls, fp_cls] += 1

    # 繪圖
    plt.figure(figsize=(12, 10))
    sns.heatmap(fp_confusion, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names,
                cmap='magma')
    plt.xlabel("Falsely Predicted Class")
    plt.ylabel("True Class")
    plt.title("False Positive Cross-Class Confusion")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("fp_cross_confusion.png")
    plt.close()
    print("Saved FP cross-class confusion matrix to fp_cross_confusion.png")


# def find_threshold_from_distribution(y_scores, y_true):
#     """
#     根據每個類別的分數，使用指定範圍的 threshold 找出最佳 F1 對應的 threshold。

#     Args:
#         y_scores (np.ndarray): 每個樣本的預測分數矩陣，形狀為 (num_samples, num_classes)。
#         y_true (np.ndarray): 每個樣本的真實標籤矩陣，形狀為 (num_samples, num_classes)。

#     Returns:
#         thresholds (list): 每個類別的最佳 threshold。
#         overall_f1 (float): 所有類別的平均 F1 Score。
#     """
#     thresholds = []
#     f1_scores_per_class = []

#     for class_idx in range(y_scores.shape[1]):
#         class_scores = y_scores[:, class_idx]
#         class_true = y_true[:, class_idx]

#         # 定義搜索範圍
#         search_range = np.arange(0.01, 1.00, 0.01)

#         # 初始化最佳 F1 分數和對應的 threshold
#         best_f1 = -1
#         best_threshold = 0

#         # 遍歷所有可能的 threshold
#         for threshold in search_range:
#             pred = (class_scores >= threshold).astype(int)
#             f1 = f1_score(class_true, pred, zero_division=0)
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold

#         thresholds.append(best_threshold)
#         f1_scores_per_class.append(best_f1)

#     # 計算 overall F1 score
#     overall_f1 = np.mean(f1_scores_per_class)

#     return thresholds, overall_f1


def find_threshold_from_distribution(y_scores, bins=100, sigma=0.1):
    """
    根據數據分布曲線，使用高斯平滑找出全域最低谷的 threshold。
    
    Args:
        y_scores (np.ndarray): 每個樣本的預測分數矩陣，形狀為 (num_samples, num_classes)。
        bins (int): 直方圖的分箱數量。
        sigma (float): 高斯平滑的標準差。
        
    Returns:
        thresholds (list): 每個類別的最佳 threshold。
    """
    thresholds = []
    for class_idx in range(y_scores.shape[1]):
        class_scores = y_scores[:, class_idx]

        # 生成直方圖
        counts, bin_edges = np.histogram(class_scores, bins=bins, range=(0.1, 0.9))

        # 使用高斯平滑
        smoothed_counts = gaussian_filter1d(counts, sigma=sigma)

        # 找到平滑後全局低谷的索引
        min_index = np.argmin(smoothed_counts)

        # 獲取對應的 threshold
        optimal_threshold = bin_edges[min_index]

        thresholds.append(optimal_threshold)

        # thresholds = [0.196,0.108,0.164,0.204,0.204,0.100,0.100,0.172,0.156,0.100,0.124,0.100,0.116,0.188,0.132] # V4 Hybrid-M
        # thresholds = [0.132, 0.124, 0.316, 0.108, 0.18, 0.108, 0.1, 0.132, 0.124, 0.1, 0.108, 0.132, 0.1, 0.14, 0.156] # V4 Conv-S
    return thresholds


# def calculate_weighted_mAP(y_true, y_scores, thresholds):
#     """
#     使用指定的 thresholds 和加權策略計算 mAP。
#     """
#     aps = []
#     weights = []
#     for class_idx in range(y_true.shape[1]):
#         threshold = thresholds[class_idx]
#         pred = (y_scores[:, class_idx] >= threshold).astype(int)
#         ap = average_precision_score(y_true[:, class_idx], pred)
#         aps.append(ap)
#         weights.append(y_true[:, class_idx].sum())  # 正樣本數作為權重
    
#     # 計算加權 mAP
#     weights = np.array(weights)
#     weighted_map = np.sum(np.array(aps) * weights) / weights.sum()
#     return weighted_map, aps



# def calculate_avg_mAP(y_true, y_scores, thresholds_range):
#     """
#     計算給定多個 thresholds 範圍內的平均 mAP。
#     """
#     avg_mAPs = []
#     for threshold in thresholds_range:
#         thresholds = [threshold] * y_true.shape[1]
#         mAP, _ = calculate_mAP_with_thresholds(y_true, y_scores, thresholds)
#         avg_mAPs.append(mAP)
#     return np.mean(avg_mAPs)


def calculate_F1_with_thresholds(y_true, y_scores, thresholds):
    """
    使用指定的 thresholds 計算每個類別的 F1 Score，並取平均。
    """
    f1_scores = []
    for class_idx, threshold in enumerate(thresholds):
        # 使用固定 threshold 進行二值化
        pred = (y_scores[:, class_idx] >= threshold).astype(int)

        # 計算該類別的 F1 Score
        f1 = f1_score(y_true[:, class_idx], pred, zero_division=0)
        f1_scores.append(f1)

    # 返回每個類別的 F1 Score 和平均 F1 Score
    return np.mean(f1_scores), f1_scores


def plot_confidence_distribution(y_scores, thresholds, labels, sigma=0.1):
    """
    為每個類別繪製 confidence 分布圖，標註最佳 threshold，並加入高斯平滑曲線。
    
    Args:
        y_scores (np.ndarray): 每個樣本的預測分數矩陣，形狀為 (num_samples, num_classes)。
        thresholds (list): 每個類別的最佳 threshold。
        labels (list): 每個類別的標籤名稱。
        sigma (float): 高斯平滑的標準差。
    """
    num_classes = y_scores.shape[1]
    bins = np.arange(0, 1.01, 0.01)  # 設置 bin 精度為 0.01
    for class_idx in range(num_classes):
        class_scores = y_scores[:, class_idx]
        
        # 計算直方圖
        counts, bin_edges = np.histogram(class_scores, bins=bins, range=(0.1, 0.8))
        
        # 高斯平滑處理
        smoothed_counts = gaussian_filter1d(counts, sigma=sigma)
        
        # 畫圖
        plt.figure(figsize=(10, 6))
        
        # 繪製直方圖，threshold 左邊部分用橘色
        colors = ['orange' if bin_edges[i] < thresholds[class_idx] else 'blue' for i in range(len(bin_edges)-1)]
        for i in range(len(counts)):
            plt.bar(bin_edges[i], counts[i], width=0.01, color=colors[i], alpha=0.7, align='edge')
        
        # 繪製高斯平滑曲線
        smoothed_x = (bin_edges[:-1] + bin_edges[1:]) / 2  # 計算 bin 中心
        plt.plot(smoothed_x, smoothed_counts, color='green', linestyle='-', linewidth=2, label='Smoothed Curve')
        
        # 標註 threshold
        optimal_threshold = thresholds[class_idx]
        plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal Threshold: {optimal_threshold:.2f}')
        
        # 標題與標籤
        plt.title(f"Confidence Distribution for Class: {labels[class_idx]}")
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.legend(loc='upper right')
        
        # 儲存圖像
        plt.savefig(f"plot_image/confidence_distribution_class_{class_idx}_{labels[class_idx]}.png")
        plt.close()  # 關閉圖像以節省記憶體
        print(f"Saved confidence distribution plot for class {class_idx}: {labels[class_idx]}")


def validate(train_loader, val_loader, model, criterion):
    """
    驗證過程：繪製 Confidence 分布、找到最佳 Threshold、計算 F1 Score。
    """
    model.eval()
    all_outputs = []
    all_targets = []

    # Collect outputs and targets for validation set
    with torch.no_grad():
        for input, target in val_loader:
            target_tensor = torch.tensor([eval(t) for t in target], dtype=torch.float32).cuda()
            _, _, output = model(input)
            # all_outputs.append(torch.softmax(output, dim=-1).cpu().numpy())
            all_outputs.append(torch.sigmoid(output).cpu().numpy())
            all_targets.append(target_tensor.cpu().numpy())

    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)

    # Collect outputs and targets for train set (for threshold finding)
    train_outputs = []
    train_targets = []

    with torch.no_grad():
        for input, target in train_loader:
            target_tensor = torch.tensor([eval(t) for t in target], dtype=torch.float32).cuda()
            _, _, output = model(input)
            # train_outputs.append(torch.softmax(output, dim=-1).cpu().numpy())
            train_outputs.append(torch.sigmoid(output).cpu().numpy())
            # print(train_outputs)
            train_targets.append(target_tensor.cpu().numpy())

    train_outputs = np.vstack(train_outputs)
    train_targets = np.vstack(train_targets)

    # Find the best threshold for each class
    best_thresholds = find_threshold_from_distribution(train_outputs) #for distribution method
    # best_thresholds,_ = find_threshold_from_distribution(train_outputs,train_targets)
    # best_thresholds,_ = find_threshold_from_distribution(all_outputs,all_targets)

    # Plot confidence distribution for each class
    labels = ['no_hand_driving', 'nap', 'seatbelt', 'Distract', 'Phone', 'Drowsy', 
              'laughing', 'single_hand_driving', 'negative', 'Singing', 'Sneezing', 
              'Eyescratching', 'Talking', 'Eating', 'Smoking']
    plot_confidence_distribution(train_outputs, best_thresholds, labels)
    pred = (all_outputs >= best_thresholds).astype(int)
    # Apply the best thresholds to make predictions
    pred = np.zeros_like(all_outputs)
    for class_idx, threshold in enumerate(best_thresholds):
        pred[:, class_idx] = (all_outputs[:, class_idx] >= threshold).astype(int)

    # Calculate precision, recall, and F1 score
    precision_per_class = precision_score(all_targets, pred, average=None, zero_division=0)
    recall_per_class = recall_score(all_targets, pred, average=None, zero_division=0)
    overall_f1, f1_per_class = calculate_F1_with_thresholds(all_targets, all_outputs, best_thresholds)

    # Save metrics for each class to Excel
    metrics_df = pd.DataFrame({
        'Class': [f'Class {i}' for i in range(len(f1_per_class))],
        'Class Name': labels,
        'Precision': np.round(precision_per_class, 2),
        'Recall': np.round(recall_per_class, 2),
        'F1 Score': np.round(f1_per_class, 2),
        'Best Threshold': best_thresholds
    })

    # Add overall metrics
    overall_metrics = pd.DataFrame({
        'Class': ['Overall'],
        'Class Name': ['Overall'],
        'Precision': [precision_score(all_targets, pred, average='macro', zero_division=0)],
        'Recall': [recall_score(all_targets, pred, average='macro', zero_division=0)],
        'F1 Score': [overall_f1],
        'Best Threshold': ['-']
    })

    metrics_df = pd.concat([overall_metrics, metrics_df], ignore_index=True)

    # Save metrics to an Excel file
    excel_filename = 'f1_score_results_with_distribution.xlsx'
    metrics_df.to_excel(excel_filename, index=False)
    print(f"F1 Score metrics for each class and overall saved to {excel_filename}")

    print(f"Overall F1 Score: {overall_f1:.4f}")

    # === For confusion matrix ===
    # 針對每筆資料找出預測的 class (multi-label -> single-label for confusion matrix)
    analyze_fp_cross_class(all_targets, pred, labels)


    return overall_f1


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def main():
    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, image_prefix, label_prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset or 'YUV' in args.modality else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    # print(model)
    
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    


    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB' or args.modality == 'YUV':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=image_prefix,
                   label_tmpl = label_prefix, 
                   transform=torchvision.transforms.Compose([
                      
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, two_stream =False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU



    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=image_prefix,
                   label_tmpl = label_prefix, 
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, two_stream =False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        # criterion = torch.nn.CrossEntropyLoss().cuda()
        criterion = torch.nn.BCEWithLogitsLoss().cuda() #for multi label
        # criterion = FocalLoss(logits=True).cuda()
    else:
        raise ValueError("Unknown loss type")
        # criterion = FocalLoss(logits=True).cuda()

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(train_loader, val_loader, model, criterion)
        return



if __name__ == '__main__':
    main()
