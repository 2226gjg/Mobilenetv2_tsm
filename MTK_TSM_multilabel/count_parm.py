import torch
from torchsummary import summary
from thop import profile
from thop import clever_format
from models_self import resnet, mobilenet

# 創建模型實例
model = resnet.resnet50(pretrained =True)
model = model.cuda()
# 創建一個虛擬輸入
dummy_input = torch.randn(1 * 8, 3, 224, 224).cuda()  # Batch size 1，每次輸入 8 個 frames

# 計算參數量和 FLOPs
flops, params = profile(model, inputs=(dummy_input,))
flops, params = clever_format([flops, params], "%.3f")

# 顯示模型的參數量和 FLOPs
print(f"Parameters: {params}")
print(f"FLOPs: {flops}")

# 使用 torchsummary 顯示模型摘要
summary(model, (3, 224, 224))  # torchsummary 無法顯示多個 frames，只能顯示單個 frame 的架構