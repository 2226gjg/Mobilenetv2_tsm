import torch
import torch.nn as nn
import torch.onnx

from ops.models import TSN
from opts import parser


class TSNWrapper(nn.Module):
    def __init__(self, model, num_segments):
        super(TSNWrapper, self).__init__()
        self.model = model
        self.num_segments = num_segments

    def forward(self, x):
        # 寫死 shape 避免 ONNX 多加推導節點
        # GRAY 模態：1 channel，故 shape 為 (8, 1, 256, 256)，對應輸入 1x8x256x256
        x = x.view(8, 1, 256, 256)
        _, base_out, output = self.model(x, no_reshape=True)
        # base_out shape: [1, 8, 15]，squeeze(0) 得到 [8, 15]（各 segment 的原始分數）
        # output shape:   [1, 15]，為 base_out 在 segment 維度做 avg consensus 的結果
        return base_out.view(1, 8, -1)


def export_to_onnx(args):
    print("==> 建立 TSM 模型架構 (GRAY 1-channel)...")
    num_class = 15
    model = TSN(
        num_class,
        args.num_segments,
        args.modality,
        base_model=args.arch,
        consensus_type=args.consensus_type,
        dropout=args.dropout,
        img_feature_dim=args.img_feature_dim,
        partial_bn=not args.no_partialbn,
        pretrain=args.pretrain,
        is_shift=args.shift,
        shift_div=args.shift_div,
        shift_place=args.shift_place,
        fc_lr5=False,
        temporal_pool=args.temporal_pool,
        non_local=args.non_local
    )

    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model.eval()

    print(f"==> 載入模型權重: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)

    model = TSNWrapper(model.module, args.num_segments).cuda()

    # GRAY 模態：1 channel × 8 segments = 8
    dummy_input = torch.randn(1, args.num_segments * 1, 256, 256).cuda()

    print(f"==> 匯出為 ONNX: {args.onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        export_params=True,
        opset_version=9,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
    )
    print("✅ 匯出完成！")


if __name__ == '__main__':
    import sys
    parser.add_argument('--checkpoint', type=str, required=True, help='path to .pth model checkpoint')
    parser.add_argument('--onnx_path',  type=str, required=True, help='path to save onnx model')
    # 對齊 run.sh 的訓練設定作為預設值
    parser.set_defaults(
        arch='mobilenetv2',
        num_segments=8,
        shift=True,
        shift_div=8,
        shift_place='blockres',
        consensus_type='avg',
        dropout=0.8,
    )
    # 固定 dataset=ucf101, modality=GRAY，使用者只需傳 optional args
    args = parser.parse_args(['ucf101', 'GRAY'] + sys.argv[1:])

    if not hasattr(args, 'gpus') or args.gpus is None:
        args.gpus = [0]

    export_to_onnx(args)
