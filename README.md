# 车道线分割项目

本项目基于 [InSPyReNet (ACCV 2022)](https://github.com/plemeri/InSPyReNet) 改造而来，当前仓库同时包含原始模型推理代码和一个可本地运行的 Flask Web 演示页面。

![Teaser](./figures/Teaser.gif)

## 项目内容

- 基于 InSPyReNet 的车道线分割模型代码
- 离线推理入口：`run/Inference.py`
- 本地网页演示入口：`app.py`
- 默认模型权重：`snapshots/HighwayLane/latest.pth`

## 1. 环境准备

建议使用 Conda 创建独立环境：

```bash
conda create -y -n lane python=3.12
conda activate lane
pip install -r requirements.txt
```

当前依赖见 `requirements.txt`，主要包括：

- `torch`
- `torchvision`
- `tqdm`
- `easydict`
- `pyyaml`
- `opencv-python`
- `thop`
- `tabulate`
- `scipy`
- `timm`
- `flask`

## 2. 权重与数据

项目默认使用的主权重路径：

```text
snapshots/HighwayLane/latest.pth
```

当前仓库内已经包含该文件。若你在新环境重新克隆仓库，还需要确认骨干网络权重存在于：

```text
data/backbone_ckpt/res2net50_v1b_26w_4s-3cf99910.pth
```

原始项目提供的数据和权重下载链接如下：

- 主模型权重：[下载链接](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EaKumdLe9iBHv1OWkjisoZ4B9ppCSvs0yZ6pxllgnGorfQ?e=9dp81y&download=1)
- 骨干网络权重：[下载链接](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EdnCVk9__w1Gh5npELiIWSIBO9DpZhHoiSLZUfGtUkwn3g?e=9zLWtn&download=1)
- 训练集：[下载链接](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfUnpxrl8jRMklEcHmp1cTcBHlQhZhSl7soRNbG0jjLb8w?e=Tj5hbe&download=1)
- 测试集：[下载链接](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/ERVEPxwzk2ZElqM7-n05COoBjcztlOnqar1bNd19tlA3Qg?e=wm6Tzh&download=1)

## 3. 命令行推理

输入可以是单张图片，也可以是一个图片目录：

```bash
python run/Inference.py --source path/to/image_or_folder
```

常用示例：

```bash
python run/Inference.py --source data/example.jpg
python run/Inference.py --source data/test_images --type overlay
python run/Inference.py --source data/test_images --dest results/custom --verbose
```

主要参数说明：

- `--source`：输入图片文件或目录
- `--dest`：输出目录，可选
- `--type`：输出类型，可选 `map`、`overlay`、`rgba`
- `--mask`：掩码图片路径，默认 `data/mask.png`
- `--config`：配置文件路径，默认 `configs/HighwayLane.yaml`
- `--verbose`：显示进度条

如果不指定 `--dest`：

- 单张图片输出到 `results/`
- 目录批量推理输出到 `results/<输入目录名>/`

## 4. 运行 Web 演示

启动本地服务：

```bash
python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

使用流程：

1. 打开首页。
2. 上传一张道路场景图片。
3. 提交推理请求。
4. 等待后端执行模型推理。
5. 在结果区域查看生成的车道线分割结果。

Web 端生成结果默认保存在：

```text
static/generated/
```

上传的临时文件默认保存在：

```text
.uploads/
```

## 5. 运行测试

执行：

```bash
pytest -q
```

## KAIST Highway 数据集性能

- 最高 F1：94.8
- 最高 IoU：88.5
- 推理速度：43 FPS
- GPU 显存占用：1.5 GB
