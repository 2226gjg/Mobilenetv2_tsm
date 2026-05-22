# 環境安裝指南


## 安裝步驟

### 1. 建立 conda 環境

```bash
conda env create -f environment.yml
conda activate mtk_tsm
```

### 2. 安裝相容版本的 setuptools

```bash
pip install "setuptools<70" "wheel<0.44"
```

### 3. 安裝 pip 套件

```bash
pip install -r requirements.txt
```

兩行都正常輸出版本即代表安裝成功。

## 啟用環境

每次使用前執行：

```bash
conda activate mtk_tsm
```
## 執行測試

```bash
sh run.sh
```