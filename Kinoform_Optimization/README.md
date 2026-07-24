# Kinoform Optimization with Dummy-Area GS Method

[![Preprint](https://img.shields.io/badge/Preprint-Jxiv-blue)](https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/5311)

[English](#english) | [日本語](#japanese)

---

<a id="english"></a>
## English

### Overview
This repository contains the C++ simulation code for generating highly optimized Phase-Only Holograms (Kinoforms). The proposed method utilizes the Gerchberg-Saxton (GS) algorithm with a dummy area and median correction to achieve both high diffraction efficiency (>80%) and high-fidelity reconstruction (PSNR >47dB).

This code is the official implementation for the paper:
**"Demonstration of High-Fidelity Reconstruction, High Diffraction Efficiency, and Data Loss Tolerance in Kinoforms"** 
by Masataka TOZUKA (Former Shonan Institute of Technology).

### Requirements
- **C++ Compiler**: Compatible with C++ 20 or later.
- **OpenCV**: Version 4.110 or later is recommended (Used for image I/O and matrix operations).
- **OpenMP**: Required to enable parallel processing for faster GS algorithm iterations.

### Directory Structure
- `Kinoform_Optimization/`: Please place it on the desktop (such as the executable file).
- `src/`: Source code (`CGH_FFT.cpp`)
- `data/`: Input image files for verification (e.g., SIT logo)
- `output/`: Directory for saving the generated CGH and reconstructed images

### About the dataset
For the large‑scale evaluation of this simulation, we use the MIT‑CGH‑4K dataset.
Because the full set of 4,000 images has a large file size, it is not included in this repository (only a few sample images for functionality checks are stored in the data/ directory).
To perform a complete reproduction of the experiments, please download the dataset from the official website below, place all images into the data/ folder, and then run the program.
- MIT-CGH-4K Dataset: https://github.com/LiangShi-MIT/TensorHolography

### Important Notes on Calculation
**Diffraction Efficiency (DE):** 
In this simulation, the physical model of the LCOS device is assumed to have an amplitude reflectance of a=1.0. When calculating the diffraction efficiency, the total sum of the intensity hologram `sum(I)` is used as the denominator, avoiding the use of `sum_input + reference_energy`.

### Citation
If you find this code useful in your research, please consider citing our preprint on Jxiv:
```text
@article{Tozuka2026,
  title={Demonstration of High-Fidelity Reconstruction, High Diffraction Efficiency, and Data Loss Tolerance in Kinoforms},
  author={Masataka Tozuka},
  journal={Jxiv},
  year={2026},
  url={ https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/5311 }
}
```

<a id="japanese"></a>
## 日本語

### 概要
本リポジトリは、高画質なキノフォーム（位相限定型ホログラム）を生成するためのC++シミュレーションコードです。ダミー領域付きGS法（Gerchberg-Saxton algorithm）と中央値補正を用いることで、高回折効率（80%超）と高忠実度再生（PSNR 47dB超）の完全な両立を実現しています。

本コードは、以下の論文の公式実装です。
**「キノフォームの高忠実度・高回折効率化およびデータ欠損耐性の実証」**
著者: 戸塚 真隆（元湘南工科大学工学部）

### 動作環境・依存ライブラリ
- **C++ コンパイラ**: C++ 20 以上
- **OpenCV**: バージョン4.110系を推奨（画像の入出力や行列計算に使用）
- **OpenMP**: GS法の反復計算を高速化するため、コンパイル時に有効化してください。

### フォルダ構成
- `Kinoform_Optimization/`: ディスクトップに置いてください（実行ファイルなど）
- `src/`: ソースコード（`CGH_FFT.cpp`）
- `data/`: 検証用の入力画像ファイル（SITロゴ画像など）
- `output/`: 生成されたCGHや再生画像の出力先

### データセットについて
本シミュレーションの大規模な評価には、MIT-CGH-4K データセットを使用しています。
データセットの全画像（4,000枚）はファイルサイズが大きいため、本リポジトリには含まれていません（動作確認用の数枚のみ `data/` に格納しています）。

完全な追試を行う場合は、以下の公式サイトからデータセットをダウンロードし、画像を `data/` フォルダ内に配置してからプログラムを実行してください。
- MIT-CGH-4K Dataset: https://github.com/LiangShi-MIT/TensorHolography

### アプリケーションの使い方


### 計算における重要な前提条件
**回折効率（DE）の算出について：**
本シミュレーションでは、LCOSデバイスの実態に合わせた物理モデル（振幅反射率 a=1.0）を前提としています。回折効率を算出する際、分母に `sum_input + reference_energy` を使用するのではなく、強度ホログラム `I(x)` の総和である `sum(I)` を分母として計算しています。追試を行う際はこの条件にご留意ください。

### 引用について
本コードを研究等で活用される場合は、Jxivにて公開中のプレプリントの引用をお願いいたします。
（※URL：https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/5311 ）