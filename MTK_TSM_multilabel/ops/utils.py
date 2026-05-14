import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, average_precision_score

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # maxk = max(topk)
    maxk = 16
    print(target)
    target_tensor = torch.tensor(target, dtype=torch.float32)
    batch_size = target_tensor.size(0)
    # batch_size = target.size(0)
    # print("maxk=",maxk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print('pred:')
    # print(pred)
    # print('target:')
    # print(target)

    ############################################
    # print(pred[:,0])

    for i in range(len(target)): ##seatbelt只跟seatbelt比
        if target[i] == 2 or target[i] == 9:
            arrange = pred[:,i]
            for j in range(len(arrange)):
                if arrange[j] == 2 or arrange [j] == 9:
                    pred[0][i] = arrange[j]
                    break

        elif target[i] == 0 or target[i] == 7: 
            arrange = pred[:,i]
            for j in range(len(arrange)):
                if arrange[j] == 0 or arrange [j] == 7 or  arrange[j] == 8:
                    pred[0][i] = arrange[j]
                    break

        elif target[i] != 2 and target[i] != 9 and target[i] != 0 and target[i] != 7:
            arrange = pred[:,i]
            # if arrange[0] == 9 or arrange[0] == 2:
            #     pred[0][i]=pred[1][i]
            for j in range(len(arrange)):
                if arrange[j] != 2 and arrange [j] != 9 and arrange [j] != 0 and arrange [j] != 7:
                    pred[0][i] = arrange[j]
                    break


    # print("update_pred=",pred)
    # pred = pred[0]
    # print("final_pred=",pred)
    ############################################
    correct = pred[0].eq(target.expand_as(pred))
    # print('correct: {}'.format(correct))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    res.append(pred[0].data)
    res.append(target.data)
    # print('res: {}'.format(res))

    
    return res


import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc

def multilabel_metrics(output, target, threshold):
    """ 計算 Precision, Recall 和 mAP for multi-label classification """
    
    # 將輸出經過 sigmoid 激活，轉換為概率
    output = torch.sigmoid(output).cpu().numpy()
    target = target.cpu().numpy()
    
    # 將概率大於閾值的轉為 1，否則為 0
    pred = (output >= threshold).astype(int)
    
    ap_scores = []

    # 計算每個類別的 AP
    for i in range(target.shape[1]):
        # 無正樣本的情況
        if np.sum(target[:, i]) == 0:
            if np.sum(pred[:, i]) > 0:  # 有偽陽性
                ap_scores.append(0.0)   # AP 設為 0
            else:
                ap_scores.append(1.0)   # 無偽陽性，AP 設為 1
        else:
            # 有正樣本的類別，使用 PR 曲線計算 AP
            precision, recall, _ = precision_recall_curve(target[:, i], output[:, i])
            ap = auc(recall, precision)  # 計算 AP
            ap_scores.append(ap)

    # 計算 Macro 平均的 AP (mAP)
    mAP = np.mean(ap_scores)

    # 計算 Precision 和 Recall
    precision = precision_score(target, pred, average='macro', zero_division=0)
    recall = recall_score(target, pred, average='macro', zero_division=0)

    return precision, recall, mAP



##不考慮沒有正樣本的mAP計算
# def multilabel_metrics(output, target, threshold):
#     """ 計算 Precision, Recall 和 mAP for multi-label classification """
    
#     # 將輸出經過 sigmoid 激活，將每個類別轉換為概率
#     output = torch.sigmoid(output).cpu().numpy()
#     target = target.cpu().numpy()
    
#     # 將概率大於閾值的轉為 1，否則轉為 0
#     pred = (output >= threshold).astype(int)
    
#     # 檢查每個類別的正樣本情況
#     # 跳過那些在 target 中完全沒有正樣本的類別
#     valid_classes = np.sum(target, axis=0) > 0  # 對於每個類別，檢查是否有正樣本
#     if not np.any(valid_classes):
#         return 0.0, 0.0, 0.0

#     # 只對那些有正樣本的類別進行指標計算
#     target_filtered = target[:, valid_classes]
#     pred_filtered = pred[:, valid_classes]
#     output_filtered = output[:, valid_classes]

#     # 計算 Precision 和 Recall，使用 zero_division 參數避免未定義問題
#     precision = precision_score(target_filtered, pred_filtered, average='macro', zero_division=0)  
#     recall = recall_score(target_filtered, pred_filtered, average='macro', zero_division=0)        
    
#     # 計算 mAP (mean Average Precision)
#     if np.sum(target_filtered) > 0:
#         try:
#             mAP = average_precision_score(target_filtered, output_filtered, average='macro')
#         except ValueError:  # 當沒有正樣本時會觸發這個錯誤
#             mAP = 0.0
#     else:
#         mAP = 0.0
    
#     return precision, recall, mAP


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, logits=False, reduction='mean'):
        """
        Args:
            alpha (float): 平衡正負樣本的權重
            gamma (float): 聚焦參數
            logits (bool): 如果輸入是 logits，設置為 True
            reduction (str): 'none', 'mean', 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        # targets = targets.unsqueeze(1)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:  # 'none'
            return F_loss