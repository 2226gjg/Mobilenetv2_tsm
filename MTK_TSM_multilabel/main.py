# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import precision_score, recall_score, f1_score
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, FocalLoss, multilabel_metrics
from ops.temporal_shift import make_temporal_pool
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

best_prec1 = 0


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
    # args.store_name = '_'.join(
    #     ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
    #      'e{}'.format(args.epochs)])
    args.store_name = 'imagenet_mean_std_20240924_resnet50_multilabel_no_freeze_lr_0.01'
    args.store_name = 'look_param'
    args.store_name = 'yen_test'
    args.store_name = 'yen_test_RGB'
    args.store_name = 'yen_test_1channel_224'
    # args.store_name = 'test'
    train_data = "/data/ivs01/MTK_TSM_multilabel/mtk_dms_20240924/mtk_dms_data_20240924_label/train_multi.txt"
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
                non_local=args.non_local,
                input_size=args.input_size)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset or 'YUV' in args.modality else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    print(model)
    
        # 計算 GFLOPS 和 Parameters
    in_ch = 1 if args.modality == 'GRAY' else (2 if args.modality == 'Flow' else 3)
    dummy_input = torch.randn(1, args.num_segments * in_ch, args.input_size, args.input_size).cuda()
    flops = FlopCountAnalysis(model, dummy_input)
    params = parameter_count_table(model)

    print(f"GFLOPS: {flops.total() / 1e9:.2f} GFLOPs")  # 轉換為 GFLOPS
    print(f"Model Parameters:\n{params}")



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
            best_prec1 = checkpoint['best_prec1']
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

        for k, v in sd.items():
            if k not in model_dict and k.replace('module.', '') in model_dict:
                print('=> Load after remove module.: ', k)
                replace_dict.append((k, k.replace('module.', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('module.', '') in sd:
                print('=> Load after adding module.: ', k)
                replace_dict.append((k.replace('module.', ''), k))




        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
            sd = {k: v for k, v in sd.items() if 'classifier' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        if args.modality == 'GRAY':
            sd = {k: v for k, v in sd.items() if 'features.0.0.weight' not in k}
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

    if args.modality == 'RGB' or args.modality == 'YUV' or args.modality == 'GRAY':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    # train_loader = torch.utils.data.DataLoader(
    #     TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
    #                new_length=data_length,
    #                modality=args.modality,
    #                image_tmpl=image_prefix,
    #                label_tmpl = label_prefix, 
    #                transform=torchvision.transforms.Compose([
    #                    train_augmentation,
    #                    Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    #                    ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    #                    normalize,
    #                ]), dense_sample=args.dense_sample),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True,
    #     drop_last=True)  # prevent something not % n_GPU

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
        # weights = [1., 1.45, 3.5, 1]

        # print(multiplier)
        # print(class_num)

        # class_weight = torch.FloatTensor(weights).cuda()
        # criterion = torch.nn.CrossEntropyLoss(weight=class_weight).cuda()

        # criterion = torch.nn.CrossEntropyLoss().cuda() #kenny use origin (before 0805 and 0805 normal ver) and start from 0918
        criterion = torch.nn.BCEWithLogitsLoss().cuda() #for multi label
        # criterion = FocalLoss(logits=True).cuda()  
    else:
        raise ValueError("Unknown loss type")
        # criterion = FocalLoss(logits=True).cuda()

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    best_threshold = 0.5  # 初始化最佳閾值

    best_prec1 = 0.0  # 用來追踪最佳 mAP
    best_epoch = 0
    best_threshold = 0.5  # 初始化最佳閾值

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch, 使用最佳閾值進行訓練
        train(train_loader, model, criterion, optimizer, epoch, log_training, best_threshold)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # 在驗證過程中計算 mAP 和最佳閾值
            mAP, best_threshold = validate(val_loader, model, criterion, epoch, log_training)

            # 假設使用 mAP 作為模型保存標準
            is_best = mAP > best_prec1  # 使用 mAP 作為比較標準

            if is_best:
                best_epoch = epoch + 1

            best_prec1 = max(mAP, best_prec1)

            output_best = 'Best mAP: %.3f, Best Threshold: %.2f\n' % (best_prec1, best_threshold)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_map': best_prec1,  # 保存最佳 mAP
                'best_threshold': best_threshold,  # 保存最佳閾值
            }, is_best, epoch + 1)

            print(f"best epoch = {best_epoch}")


# 修改 find_best_threshold 函數，計算並返回最佳閾值
def find_best_threshold(output, target, thresholds=np.arange(0.1, 0.9, 0.1)):
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        pred = (output >= threshold).astype(int)
        precision = precision_score(target, pred, average='macro', zero_division=0)
        recall = recall_score(target, pred, average='macro', zero_division=0)
        f1 = f1_score(target, pred, average='macro', zero_division=0)
        
        print(f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best Threshold: {best_threshold}, Best F1: {best_f1:.4f}")
    return best_threshold

# 修改 validate 函數，將其與 find_best_threshold 集成
def validate(val_loader, model, criterion, epoch, log=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    mAPs = AverageMeter()

    model.eval()

    end = time.time()

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = [eval(t) for t in target]
            target_tensor = torch.tensor(target, dtype=torch.float32).cuda()

            input_for_heatmap, base_out, output = model(input)

            if args.consensus_type == 'avg':
                base_out = base_out.mean(dim=1, keepdim=True)
            base_out = base_out.squeeze(1)

            loss = criterion(output, target_tensor)

            # 收集所有的輸出和目標，用於後續找最佳閾值
            all_outputs.append(torch.sigmoid(output).cpu().numpy())
            all_targets.append(target_tensor.cpu().numpy())

            losses.update(loss.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    # 將所有 batch 的輸出拼接在一起
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)

    # 在驗證集上計算最佳閾值
    best_threshold = find_best_threshold(all_outputs, all_targets)

    # 計算 mAP, Precision, Recall 等指標
    precision, recall, mAP = multilabel_metrics(torch.tensor(all_outputs), torch.tensor(all_targets), threshold=best_threshold)

    output = ('Testing Results: Best Threshold {best_threshold:.2f}, mAP: {mAP:.4f}, Loss: {loss.avg:.5f}'.format(
              best_threshold=best_threshold, mAP=mAP, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    return mAP, best_threshold  # 返回 mAP 和最佳閾值


# 修改 train 函數，使用最佳閾值進行訓練
def train(train_loader, model, criterion, optimizer, epoch, log, best_threshold):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    mAPs = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # 將 target 轉換為多標籤張量，並將其放到 GPU 上
        target = [eval(t) for t in target]
        target_tensor = torch.tensor(target, dtype=torch.float32).cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target_tensor)

        # 打印 y_true (target)
        # print(f"Epoch {epoch} Batch {i}: y_true (target):\n{target_tensor}")    

        # compute output
        input_for_heatmap, base_out, output = model(input_var)

        if args.consensus_type == 'avg':
            base_out = base_out.mean(dim=1, keepdim=True)
        base_out = base_out.squeeze(1)


        loss = criterion(output, target_var)

        # 計算 Precision, Recall, 和 mAP
        precision, recall, mAP = multilabel_metrics(output.data, target_tensor, threshold=best_threshold)
        precisions.update(precision, input.size(0))
        recalls.update(recall, input.size(0))
        mAPs.update(mAP, input.size(0))

        losses.update(loss.item(), input.size(0))

        loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'mAP {mAP.val:.3f} ({mAP.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, prec=precisions, 
                recall=recalls, mAP=mAPs, lr=optimizer.param_groups[-1]['lr'] * 0.1))
                # recall=recalls, mAP=mAPs, lr=optimizer.param_groups[-1]['lr']))
                # recall=recalls, mAP=mAPs, lr=0.05))
            print(output)
            log.write(output + '\n')
            log.flush()

def save_checkpoint(state, is_best, epoch):
    # 構建檔案名稱，包含 epoch 數
    if is_best:
        filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
        torch.save(state, filename)

    elif (epoch%50 == 0) :
        filename = '%s/%s/ckpt_epoch_%d.pth.tar' % (args.root_model, args.store_name, epoch)
        torch.save(state, filename)

    
    
   


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
