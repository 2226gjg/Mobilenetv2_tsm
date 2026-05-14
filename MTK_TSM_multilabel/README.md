# MTK TSM Multilabel — 灰階影片多標籤辨識

以 **MobileNetV2 + TSM（Temporal Shift Module）** 為骨幹的多標籤影片動作辨識模型。  
輸入為單通道影片幀序列，輸出每個類別的預測分數。

---

## 目錄

1. [環境需求](#環境需求)
2. [專案結構](#專案結構)
3. [Model Input / Output Shape](#model-input--output-shape)
4. [快速開始（使用內建 Demo 資料）](#快速開始使用內建-demo-資料)
5. [使用自訂資料集](#使用自訂資料集)
6. [訓練設定說明](#訓練設定說明)

---

## 環境需求

```
依照README_ENV.md 安裝conda 環境
```


---

## 專案結構

```
MTK_TSM_multilabel/
├── annotation/                  ← 資料集標注（demo 資料已內建）
│   ├── classInd.txt             ← 類別定義（15 類）
│   ├── train.txt                ← 訓練清單
│   ├── val.txt                  ← 驗證清單
│   └── frames/                  ← 影片幀資料夾
│       ├── no_hand_driving/
│       ├── nap/
│       └── ...
├── pretrained/                  ← 預訓練權重（Kinetics RGB MobileNetV2）
├── checkpoint/                  ← 訓練產生的模型權重（自動建立）
├── log/                         ← 訓練 log（自動建立）
├── ops/
│   ├── models.py                ← TSN 模型定義
│   ├── dataset.py               ← 資料讀取
│   ├── dataset_config.py        ← 資料集路徑設定 ← 主要修改這裡
│   └── transforms.py            ← 資料前處理
├── archs/
│   └── mobilenet_v2.py          ← MobileNetV2 架構
├── main.py                      ← 訓練主程式 ← store_name 在這裡設定
└── run.sh                       ← 訓練啟動腳本 ← batch size / epochs 在這裡設定
```

---

## Model Input / Output Shape

### 訓練 / 推論時的 Tensor

| 項目 | Shape | 說明 |
|---|---|---|
| **Input** | `[B, 8, 256, 256]` | B = batch size，8 segments，單通道灰階，256×256 |
| **Output** | `[B, 15]` | 15 個類別的原始分數（sigmoid 前），multi-label |

> 模型內部會將 `[B, 8, 256, 256]` reshape 為 `[B×8, 1, 256, 256]` 送入 2D conv，  
> 再透過 TSM 的 temporal shift 在 segments 之間傳遞時序資訊，最後 avg pooling 合併成一個預測。

---

## 快速開始（使用內建 Demo 資料）

內建 `annotation/` 已包含 30 筆 train、15 筆 val，涵蓋全部 15 個類別，**不需要任何路徑設定即可直接執行**。

### Step 1：確認預訓練權重存在

```
pretrained/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth
```

### Step 2：執行訓練

```bash
sh run.sh
```

訓練結果輸出：
- 模型權重：`checkpoint/yen_test_cos/ckpt.pth.tar`
- 訓練 log：`log/yen_test_cos/log.csv`

---

## 使用自訂資料集

### Step 1：準備資料夾結構

影片幀需整理成以下格式（每個影片一個資料夾，內含連續幀圖片）：

```
your_frames_root/
├── class_A/
│   ├── video_001/
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   └── ...
│   └── video_002/
└── class_B/
    └── ...
```

### Step 2：準備 txt 標注檔

`train.txt` / `val.txt` 每行格式：

```
class_A/video_001 幀數 [1, 0, 1, 0, ...]
```

- 第一欄：相對於 `frames/` 根目錄的路徑
- 第二欄：該影片總幀數（需 ≥ 8）
- 第三欄：長度為 15 的 multi-label 向量

### Step 3：更新路徑設定

編輯 `ops/dataset_config.py`，修改 `_ANNOTATION` 指向你的資料夾：

```python
# ops/dataset_config.py 第 10 行
_ANNOTATION = os.path.join(_PROJECT_ROOT, 'annotation')   # ← 改成你的路徑
```

或直接修改 `return_ucf101` 內的三條路徑：

```python
root_data          = os.path.join(_ANNOTATION, 'frames')       # 幀資料根目錄
filename_imglist_train = os.path.join(_ANNOTATION, 'train.txt') # 訓練清單
filename_imglist_val   = os.path.join(_ANNOTATION, 'val.txt')   # 驗證清單
```

---

## 訓練設定說明

### `run.sh` — 常用訓練參數

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py ucf101 GRAY \
     --arch mobilenetv2 --num_segments 8 \
     --gd 10 --lr 0.01 --lr_steps 10 20 --epochs 300 \
     --batch-size 8 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --lr_type cos\
     --tune_from=pretrained/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth

```

| 參數 | 說明 |
|---|---|
| `CUDA_VISIBLE_DEVICES` | 使用的 GPU 編號 |
| `--epochs` | 訓練總 epoch 數 |
| `--eval-freq` | 每幾個 epoch 做一次 validation |

### `main.py` — checkpoint / log 命名

```python
# main.py 第 46 行
args.store_name = 'yen_test'   # ← 改成你想要的實驗名稱
```

- 權重儲存至：`checkpoint/{store_name}_cos/ckpt.pth.tar`
- Log 儲存至：`log/{store_name}_cos/log.csv`

---

## 類別說明

`annotation/classInd.txt` 定義 15 個類別（index 0–14），對應 output 向量的各個位元，  
各類別實際語意請參照資料標注定義。

