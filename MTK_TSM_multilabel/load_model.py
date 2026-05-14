import torch

# 讀取.pth檔
data = torch.load('/home/3105825/nova_action/checkpoint/best_top1_acc_epoch_65.pth')
# data = torch.load('/home/3105825/nova_action/TSM_kinetics_RGB_resnet18_shift8_blockres_avg_segment8_e50/new_ckpt.best.pth')

print(data['state_dict'].keys())

# ----------------------------------------------------------------
# 判斷存儲的數據類型並打印
# if isinstance(data, dict):
#     for key, value in data.items():
#         print(f"Key: {key}, Value Type: {type(value)}")
# elif isinstance(data, torch.nn.Module):
#     print("This .pth file contains a PyTorch model.")
# else:
#     print(f"The loaded data type is: {type(data)}")

# --------------------------------------------------------------------
if 'state_dict' in data and isinstance(data['state_dict'], dict):
    keys_to_delete = [key for key in data['state_dict'].keys() if 'batches' in key or 'temp' in key]
    for key in keys_to_delete:
        del data['state_dict'][key]

    # 儲存更新後的數據
    torch.save(data, '/home/3105825/nova_action/checkpoint/best_top1_acc_epoch_65.pth')
else:
    print("The loaded data does not appear to have a 'state_dict' key or it's not a dictionary.")
# --------------------------------------------------------------------