# 本地 Git 版本管理

本仓库位于 `C:/Users/47348/Desktop/AIC2/qw-advice`。Git保存代码与小型实验依据，数据集、权重、预测结果和压缩包继续保留在原目录，未纳入Git。

## 当前版本关系

- `main`：接入Git时的基线代码快照，保留原训练入口、监控入口及已有验证工具。
- `baseline-before-class-fp`：上述快照的固定标签。
- `experiment/class-fp`：最新类别误报实验代码，修改CE类别权重并增加precision/recall监控。代码检查通过，正式70轮效果尚待验证。
- `class-fp-v1`：本次类别误报实验代码的固定标签。

“main基线”指接入Git时保存的代码状态，不是宣称整个目录逐文件等于历史服务器运行版本。历史运行中的源码哈希和参数仍以原监控记录为准。

## 现在用哪个入口

| 用途 | 入口 |
|---|---|
| 本轮类别误报实验 | `bash prove/class_fp/run_ablation.sh inverse_sqrt` |
| 类别权重训练入口 | `prove/class_fp/train_fp.py` |
| 原监控训练入口 | `prove/training_monitor/train_monitored.py` |
| 预测与提交校验 | 根目录 `predict.py`、`validate_submission.py` |
| BN校准实验 | `prove/bn_calibration/`，仅保留复核证据；校准后成绩下降，未采用 |

当前实验不自动替换根目录 `train.py`。正式训练验证通过后，再将需要保留的改动合入稳定代码。查看类别误报实验说明：`prove/class_fp/README.md`。

## 日常改动

可以直接编辑当前版本中的源文件。Git会保存提交历史，之后不必另存“code_old”“code_continue”等目录来充当版本管理。

```bash
git status --short
git diff
git add -- path/to/changed_file.py
git commit -m "Describe the concrete change"
```

每个独立改动单独提交。不要把本轮类别权重与下一轮跨场景增强混成同一个实验，否则难以判断收益来源。

正式训练前，记录使用的提交编号：

```bash
git rev-parse HEAD
git status --short
```

如有未提交的源代码改动，先检查并提交再开始正式实验。现有监控还会记录训练参数、固定划分和源码SHA256。

## 查看与比较历史

```bash
git log --oneline --decorate --all
git diff baseline-before-class-fp..class-fp-v1 --stat
git show class-fp-v1:prove/class_fp/train_fp.py
```

先查看或比较历史；只有确实需要切换代码时才切分支。切换分支可能增删该版本所跟踪的源文件，应先保存当前改动并结束依赖这些源文件的训练进程。忽略的权重和数据不是Git备份的一部分。

## 给远程服务器打包

当前源码的版本依据是Git提交。可以从已提交版本导出，不再手工拼接不同旧压缩包：

```bash
git archive --format=zip --output=source-class-fp-v1.zip class-fp-v1
```

在新的项目目录解压，再使用原数据路径；数据和模型权重仍单独管理。该压缩包只包含被Git跟踪的文件。本仓库目前仅保存在本机，没有配置远程仓库，也没有上传GitHub。

## 跟踪范围与字节一致性

`.gitignore`排除了数据、权重、运行日志、缓存、合成测试输出、解压测试目录和ZIP；小型固定划分、类别权重依据、BN实验结论保留在版本中。需要新增实验目录时，应明确将对应源代码和配置加入规则，不要直接把全部运行产物放入Git。

仓库级 `core.autocrlf=false`，并通过 `.gitattributes`保留既有源文件字节，避免提交过程改变历史实验的源码哈希。Linux shell脚本固定使用LF。

