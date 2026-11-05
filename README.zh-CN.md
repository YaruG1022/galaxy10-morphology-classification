# 星系形态分类（Galaxy Morphology Classification）

[English](README.md)

使用 Galaxy10 DECaLS 的十类星系图片，比较基于手工特征（handcrafted features）的传统分类方法、从头训练的卷积神经网络（Convolutional Neural Network, CNN），以及预训练卷积神经网络（pretrained CNNs）和视觉变换器（vision transformers）。

**当前进度：数据准备、探索性数据分析（Exploratory Data Analysis, EDA）和手工特征工程（handcrafted feature engineering）已完成；分类模型训练（classifier training）与性能评价（performance evaluation）尚未开始。** 原始文件有 17,675 张图片，清理后保留 17,655 张。前三个模块的记录、图表和已执行的交互式笔记本（notebooks）已保存在本项目中。

## 研究问题

- 从基于手工特征（handcrafted features）的传统模型（classical models）到从头训练的卷积神经网络（CNN）和微调的预训练模型（fine-tuned pretrained models），分类效果会怎样变化？
- 对于受扰动星系（Disturbed）和并合星系（Merging），视觉变换器（vision transformers）的召回率（recall）是否高于规模相近的卷积神经网络（CNN）？
- 不同模型容易混淆哪些类别和图片？
- 按类别加权的训练（class-weighted training）能否改善少数类别的表现？

## 计划比较的方法

| 类型 | 方法 |
| --- | --- |
| 基于手工特征（handcrafted features） | 多项逻辑回归（multinomial logistic regression）、k 近邻（k-nearest neighbors, kNN）、径向基函数支持向量机（Radial Basis Function Support Vector Machine, RBF SVM）、随机森林（random forest）、极端梯度提升（Extreme Gradient Boosting, XGBoost） |
| 从头训练的卷积神经网络（CNN） | 包含五个卷积块（convolutional blocks）的自定义网络 |
| 预训练卷积神经网络（pretrained CNNs） | ResNet-50、ConvNeXt-Tiny |
| 预训练视觉变换器（pretrained vision transformers） | ViT-Small/16、Swin-Tiny |

另设始终预测最多类别的基准方法（majority-class baseline），用于检查结果，不计入上述十种方法。

## 已完成的三个模块

### 1. 数据准备（data preparation）

核对原始文件的校验值（checksum）、图片尺寸、数据类型（data types）及类别标注（class labels），检查异常图片和重复图片，并保存排除原因。

共排除 20 张图片：1 张所有像素值相同的图片、17 张冗余截图，以及 2 张相互重叠但类别标注冲突的图片。最终保留 17,655 张。原始文件保持不变，排除决定记录在样本清单（sample manifest）中。重复筛查有范围限制，不能保证所有重复情况均已消除。

所有方法必须使用同一份固定数据划分（data split）：

| 用途 | 图片数 | 原始行号文件（original-row index file） |
| --- | ---: | --- |
| 训练集（training set） | 12,359 | [train_indices.json](metadata/data_v1/train_indices.json) |
| 验证集（validation set） | 2,650 | [validation_indices.json](metadata/data_v1/validation_indices.json) |
| 测试集（test set） | 2,646 | [test_indices.json](metadata/data_v1/test_indices.json) |

比例约为 70/15/15，随机种子（random seed）为 42。上述文件保存的是从 0 开始的原始行号（original row indices），图片仍从同一个原始文件读取，没有复制成三个图片文件夹。

完整记录见[数据说明](docs/DATA.md)，逐张图片的保留、排除及划分记录见 [sample_manifest.csv](metadata/data_v1/sample_manifest.csv)。

### 2. 探索性数据分析（EDA）

已生成七张图，涵盖类别数量、每类图片示例、类别平均图片、像素强度、图片通道差异，以及红移（redshift）和像素尺度（pixel scale）。所有基于图片的统计和示例选择只使用训练集（training set）；全体类别数量来自固定的样本清单（sample manifest）。每类使用固定随机种子（random seed）选取四张示例图片。

最多的类别有 2,640 张图片，最少的只有 333 张，相差约 7.93 倍。这种类别不平衡（class imbalance）意味着整体准确率（accuracy）可能掩盖少数类别的问题。图片中的邻近天体、背景和目标大小差异，也可能影响整张图片的统计值。

这些图描述数据特点，没有给出分类性能结论。记录见[探索性分析](docs/eda/README.md)，执行过程见 [01_exploration.ipynb](notebooks/01_exploration.ipynb)。

### 3. 手工特征工程（handcrafted feature engineering）

已为全部 17,655 张保留图片提取颜色统计、形态代理量（morphology proxies）、形状描述量（shape descriptors）、质量标记（quality flags），以及方向梯度直方图（Histogram of Oriented Gradients, HOG）。

每张图片有 54 个数值及质量描述量、1,764 个原始纹理特征（texture features）。缺失值填补（imputation）、标准化（standardization）及主成分分析（Principal Component Analysis, PCA）只从训练数据学习处理参数，将纹理部分压缩为 64 个分量，最终得到每张图片 118 个输入特征（input features）。

- 类别标注（class labels）、原始行号（original row indices）、红移（redshift）、像素尺度（pixel scale）和天空坐标（sky coordinates）均不作为输入特征（input features）。
- 训练图片中没有出现无法计算的数值描述量；2,082 张图片的目标掩膜（target masks）触及预设检测边界，占 16.85%。
- 固定的 64 个主成分（principal components）保留了标准化纹理数据方差（variance）的 42.03%。这是预先固定的设置，尚未通过分类实验确定效果。
- 形态代理量（morphology proxies）来自显示合成图（display composites），不是经过天文标定的物理测量。
- 训练特征的统一流形逼近与投影（Uniform Manifold Approximation and Projection, UMAP）只用于观察二维分布，没有使用类别标注（class labels）参与拟合（fitting），不能作为分类性能证据。预训练表示（pretrained representations）的统一流形逼近与投影（UMAP）尚未完成。

记录见[特征说明](docs/features/README.md)、[固定配置](configs/features_v1.json)和 [02_handcrafted_features.ipynb](notebooks/02_handcrafted_features.ipynb)。

## 后续评价规则

以宏平均 F1 分数（macro-averaged F1 score）作为主要指标，并报告各类别召回率（per-class recall）。测试集（test set）保留到方案确定后进行最终评价；深度模型（deep models）计划各运行三个随机种子（random seeds），具体取值尚未确定。

后续交叉验证（cross-validation）必须将**原始描述量（raw descriptors）**传给 `galaxy_features.py` 中的 `make_model_pipeline(estimator)`，并将整条处理流水线（pipeline）放入 `GridSearchCV`。这样，每个训练折（training fold）都会重新学习填补、标准化和降维参数，避免数据泄漏（data leakage）。不要直接用已按完整训练集处理过的特征进行内部交叉验证（inner cross-validation）。

已保存的完整训练集预处理器（preprocessor）用于最终训练及对留出数据的推断（held-out inference）。详细规则见[实验计划](docs/EXPERIMENT_PLAN.md)。

## 在本地运行

使用 Python 3.12。先从 [Zenodo](https://zenodo.org/records/10845026) 下载 `Galaxy10_DECals_NoDuplicated.h5`，放入 `data/raw/`。以下命令均在项目根目录执行。

检查原始文件与已保存的数据划分（data split）：

```powershell
python -m pip install -r requirements-data.txt
python check_data.py
python validate_data_preparation.py
```

`check_data.py` 检查未修改的原始文件，因此仍会报告已记录的原始问题。`validate_data_preparation.py` 检查清理记录、原始文件校验值（checksum）和固定数据划分（data split），包括是否完整、是否互不重叠，以及已知重复候选（duplicate candidates）是否跨越不同划分。

重新生成探索性分析（EDA）：

```powershell
python -m pip install -r requirements-eda.txt
python eda.py
python validate_eda.py
```

提取并检查手工特征（handcrafted features）：

```powershell
python -m pip install -r requirements-features.txt
python build_features.py
python validate_features.py
```

运行现有检查用例（test cases）：

```powershell
python -m unittest discover -s tests -v
```

当前机器已准备好项目虚拟环境（virtual environment），也可使用 `.venv\Scripts\python.exe`。特征提取（feature extraction）会处理全部保留图片，耗时取决于机器配置。

`prepare_data.py` 用于重建筛查和清理过程，默认拒绝覆盖已经完成的固定版本。若要重新检查，应使用新的输出和缓存目录（cache directories），先执行 `--scan-only`，记录复核决定后再执行 `--finalize`；具体说明见[数据记录](docs/DATA.md)。分类模型训练命令将在对应模块实现后添加。

## 文件位置

| 路径 | 内容 |
| --- | --- |
| `data/raw/` | 原始图片文件，仅保存在本地 |
| `metadata/data_v1/` | 固定数据划分（data split）、样本清单（sample manifest）、清理决定及校验记录 |
| `data/processed/preparation_v1/` | 本地缩略图缓存（thumbnail cache）及图片复核表 |
| `docs/eda/` | 探索性分析（EDA）的图表、统计表和解释 |
| `data/processed/eda_v1/` | 本地逐张图片诊断（per-image diagnostics）及缓存数组（cached arrays） |
| `docs/features/` | 特征提取（feature extraction）的方法、诊断图和限制 |
| `metadata/features_v1/` | 特征定义、处理配置、拟合行号（fit indices）及校验记录 |
| `data/processed/features_v1/` | 本地特征矩阵（feature matrices）及预处理器（preprocessor） |
| `notebooks/` | 已执行的交互式笔记本（notebooks） |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | 计划方法、评价指标（metrics）、目标和待定事项 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 后续阶段和计划分工 |
| [docs/ENVIRONMENT_PLAN.md](docs/ENVIRONMENT_PLAN.md) | 运行环境和计算资源计划 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 协作与实验记录约定 |
| [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE) | 任务和实验模板 |
| [.gitignore](.gitignore) | 本地环境、密钥、数据、模型产物（model artifacts）及 `AGENTS.md` 的忽略规则（ignore rules） |

`metadata/data_v1/` 中的 `review_decisions.json` 记录排除决定和复核来源；`summary.json` 记录数量、随机种子（random seed）、文件哈希值（file hashes）和筛查限制。

`data/processed/features_v1/` 中的 `raw_features.npz` 保存原始描述量（raw descriptors）；`train_features.npz`、`validation_features.npz`、`test_features.npz` 保存各部分已经处理的 118 维特征（features）。类别和原始行号另存，不包含在输入数组 `X` 中。标量特征表（scalar feature table）中的 `original_index`、`label` 和 `split` 同样只是附带信息，不是模型输入。

原始数据、大型矩阵和本地运行环境由 `.gitignore` 排除，不会随代码上传。小型配置、记录和整理后的图表保存在版本控制（version control）中。

## 团队分工

以下是计划职责，不代表所有工作已经完成：

- **Yaru Gao：** 数据获取、清理、重复检查、固定数据划分（data split）、探索性分析（EDA）和手工特征（handcrafted features）。
- **Zhongyan Zhang：** 传统模型（classical models）、参数调优（hyperparameter tuning）和统一评价。
- **Zijian Zhang：** 深度模型（deep models）、计算任务和可解释性分析（interpretability analysis）。
- **全体成员：** 错误分析（error analysis）、研究报告和展示。

分工可随项目进展调整。GitHub 协作者邀请尚未配置。

## 数据来源与许可

数据来自 Leung 和 Bovy 整理的 [Galaxy10 DECaLS](https://zenodo.org/records/10845026)，图片来自 DESI Legacy Imaging Surveys，类别标注（class labels）来自 Galaxy Zoo。数据采用知识共享署名 4.0 国际许可（Creative Commons Attribution 4.0 International, CC BY 4.0）；说明见[数据记录](docs/DATA.md)。项目尚未选择软件许可证（software license）。

本项目使用的去重版文件有 17,675 张图片，不能套用原始发布版的 17,736 张；清理后用于比较的是 17,655 张。

研究计划以团队 2026 年 10 月 7 日修订的提案为基础，原始材料保留在本地，不会上传。协作和工作来源记录见 [CONTRIBUTING.md](CONTRIBUTING.md)；团队需要理解并独立核实数据准备决定和分析解释。
