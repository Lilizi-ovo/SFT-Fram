# Fram：Qwen3.5-9B 全量 SFT 与验证

面向单机八卡 RTX PRO 6000、约两万条数据，使用 **Python 控制入口 + ms-swift 4.x + Bash 参数文件**。路径均相对于项目目录解析，迁移服务器不需要修改 Python 源码。

## 目录

```text
Fram/
├── scr/                 Python 控制逻辑（沿用参考图的 scr 命名）
├── sh/                  已配置的原生 CoT 训练和验证参数
├── data/                服务器部署训练/测试集与媒体的位置
├── model/               你准备的基础模型，当前留空
├── outputs/             训练权重、日志、验证结果，当前留空
├── cache/               框架与依赖的运行缓存
├── docs/parameters.md    训练和验证参数文件的完整示例
├── tools/               路径缓存、真实编码预检、分层评估工具
├── tests/               不依赖 CUDA 的流程与评分测试
├── main.py              Python 入口
├── st.bat               Windows 交互终端入口
├── st.sh                Linux 入口
└── requirements.txt     服务器 Python 依赖
```

`sh/` 已提供与当前数据对应的配置。数据、权重、运行缓存与 outputs 不提交 Git；部署后按 [参数文档](docs/parameters.md) 重建服务器绝对图片路径缓存。

## 在服务器安装

推荐 **Linux + Python 3.12**。先保留并检查服务器现有的驱动/CUDA/PyTorch 环境，再决定是否新建环境；不要复制本机 Windows 的虚拟环境到服务器。

```bash
cd /你的服务器路径/Fram
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

先依据 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装匹配服务器的 CUDA 版 `torch`、`torchvision`。RTX PRO 6000 的具体代际以 `nvidia-smi` 为准：若为 Blackwell，需要支持其架构的 CUDA/PyTorch 构建，不要沿用旧卡环境。然后执行：

```bash
python -m pip install -r requirements.txt
python main.py doctor
```

Qwen3.5 使用混合注意力。为改善 GatedDeltaNet 路径的训练性能，可在已经装好 PyTorch 和 CUDA 编译环境后安装：

```bash
python -m pip install 'flash-linear-attention>=0.4.2' --no-build-isolation
python -m pip install causal-conv1d --no-build-isolation
python -m pip check
```

这些扩展需要匹配服务器环境，因此没有让本机启动器自动安装。示例采用 `sdpa`，不强制安装 FlashAttention。不要把本仓库的依赖范围当作已完成八卡验证的锁定组合；首次服务器冒烟测试通过后，使用 `python -m pip freeze > environment.lock.txt` 固定实际环境。每次运行也会记录核心依赖版本和 GPU 信息。

## 准备输入

将完整的 Qwen3.5-9B 模型放入 `model/Qwen3.5-9B/`，包括模型配置、权重、tokenizer、processor 和聊天模板文件，不要只拷贝权重分片。模型也可使用远程模型 ID，在参数文件里设置 `--use_hf true` 切换下载源；默认配置使用本地模型。

当前使用清理版 27,037 条原生 CoT 训练数据与固定 3,000 条测试数据，配置已在 `sh/` 中建立，见 [参数文件和准备命令](docs/parameters.md)。本项目使用参数数组式 `.sh`：文件只设置环境变量与 `SWIFT_ARGS=(...)`，不直接调用 `swift sft`。读取配置时会执行该文件，**只使用自己信任的参数文件**。`--dry-run` 同样需要读取配置；它不会启动 Swift，但不是任意 Shell 代码的沙箱。

数据使用 Swift 的 `messages` JSONL 格式。训练 assistant 应包含非空推理与裸答案，例如：

```json
{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"<think>Checkable clue-to-answer explanation in English.</think>\n\n答案词"}]}
```

图片示例：

```json
{"messages":[{"role":"user","content":"<image>题目及类别、字数提示"},{"role":"assistant","content":"<think>Checkable clue-to-answer explanation in English.</think>\n\n答案词"}],"images":["data/images/0001.jpg"]}
```

运行工作目录始终是 Fram 根目录，因此媒体路径写成 `data/images/0001.jpg`。不要使用本机盘符或仅相对于 JSONL 文件位置的路径。验证集的 `assistant` 内容是参考答案，Swift 在数据集推理时用于对照；训练集、验证集请预先分开，避免同一图片或同一问题泄漏。

## 使用方式

Windows 双击 `st.bat`，或在 Linux 执行 `bash st.sh`，进入 `fram>` 交互终端后输入：

```text
configs
train -full-SFT --dry-run
train -full-SFT
runs
val -20261019 -val --dry-run
val -20261019 -val
exit
```

`train -full-SFT` 读取 `sh/full-SFT.sh`。`val -20261019 -val` 从 `outputs/20261019` 中选择模型，并读取 `sh/val.sh`。开头的单横线保留了参考图的输入习惯；`train full-SFT`、`val 20261019 val` 同样有效。

也可直接调用，便于 `tmux`、Slurm 或任务调度器使用：

```bash
python main.py train -full-SFT --dry-run
python main.py train -full-SFT --name experiment-01
python main.py val -experiment-01 -val
python main.py val -experiment-01 -val --checkpoint checkpoint-500
```

训练输出默认是当天的 `outputs/YYYYMMDD/`，同一天的新训练自动使用 `YYYYMMDD-02` 等后缀；显式 `--name` 遇到已有目录会拒绝覆盖。图中的 `20261019` 是目录名示例，需要替换成实际训练输出。目录中将包含：

```text
outputs/experiment-01/
├── run.json                 运行状态、命令、退出码、核心环境信息
├── parameters.sh            本次顶层参数文件快照
├── requirements.freeze.txt  本次核心依赖版本（不是全部传递依赖的锁文件）
├── console.log              合并后的标准输出和错误日志
├── checkpoint-*/            Swift 保存的完整模型与训练状态
└── eval/YYYYMMDD-HHMMSS/    每次验证独立保存，避免追加混淆
    ├── run.json
    ├── console.log
    ├── predictions.jsonl    Swift 原生逐条推理结果
    ├── eval_manifest.jsonl  该次冻结的答案词分层清单
    ├── stratified_metrics.json  整体、已见/未见与题型成绩
    └── stratified_metrics.details.jsonl  逐题评分依据
```

配置文件中不要设置 `--output_dir`、`--add_version`、`--result_path` 或 `--resume_from_checkpoint`，这些参数由入口统一管理。验证时 `--model`、`--adapters` 也由入口管理。训练禁用 Swift 自动增加的 `v*-时间` 子目录，使图中的目录形式保持可用。若使用旧版外部训练结果，其目录中有 `v*-时间` 层，可直接指定 `val -旧运行/v0-实际时间 -val`。

默认按 checkpoint 的**数字步数**选择最新完整模型，不按文件修改时间排序。缺少模型配置或分片的 checkpoint 不会被选中；如果有更新但尚未保存完整的 checkpoint，会使用之前的完整 checkpoint，因此不要在训练仍写入权重时运行验证。可用 `--checkpoint` 明确指定。只有 ZeRO 原始分片或 LoRA adapter 的目录不能直接作为本项目的完整模型验证输入。

## 断点恢复与停止

```bash
python main.py train -full-SFT --resume experiment-01/checkpoint-500 --name experiment-01-resume
```

完整恢复需要 checkpoint 中的优化器、随机数等训练状态；示例设置 `--save_only_model false`。`--resume experiment-01` 会选择完整模型中步数最大的 checkpoint。续训输出始终写到一个新目录，不覆盖原来的日志和权重；框架保留原 `--model` 基础模型参数，并把 checkpoint 交给 Swift 的 `resume_from_checkpoint` 恢复。

按 Ctrl+C 或接收 SIGTERM 会结束当前训练/验证进程组，并尽可能将状态记录为 `interrupted`；退出码分别为 130、143。交互终端保留。通过单次命令调用时，非零退出码会返回给调度器。断电或 `kill -9` 无法更新最后状态，`run.json` 中的 `running` 不代表进程仍存活。

## 服务器第一次运行

1. 检查 `doctor` 是否显示正确的 Python、依赖和八张可见 GPU。
2. 运行数据准备工具和实际 Swift 编码预检，再用 `--dry-run` 检查命令。默认 thinking=true、max_length=8192、IMAGE_MAX_TOKEN_NUM=1024。
3. 用少量训练样本并保留 `max_length=8192`，临时加入 `--max_steps 2 --save_steps 1` 做一次真实训练/保存/验证冒烟测试；通过后移除临时参数。最终测试不能用于反复选 checkpoint。

约两万条数据不能单独决定合适的 batch size 或训练耗时，文本长度、图片数量、分辨率和磁盘吞吐都会影响显存与速度。示例有效 batch size 为 `8 卡 × 每卡 1 × 累积 4 = 32`，只是保守起点。全量多模态训练显式设置 `freeze_llm/vit/aligner=false`；若只训练语言部分，可以在参数文件中修改冻结策略。

验证入口开启 thinking，并自动按最终答案评分，输出已见/未见答案词、题型与交叉分组成绩。原词和规范化两种口径同时保留；不使用对整段推理做相等比较的 Swift 内置 acc。没有模型预测时只报告分层样本数。

## 开发检查与参考

控制层只用标准库，单元与集成测试不下载模型、不占用 GPU：

```bash
python -m unittest discover -s tests -v
```

测试使用临时目录，覆盖 Bash 读取、带空格路径、checkpoint 分片检查、输出隔离、假 Swift 子进程的成功/失败/中断以及交互式命令解析。CI 在 Linux 和 Windows 的 Python 3.12 上运行相同检查。

实现依据（2026-10-27 核对）：[Swift 命令参数](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)、[Qwen3.5 最佳实践](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Best-Practice.html)、[Qwen3.5-9B 模型](https://huggingface.co/Qwen/Qwen3.5-9B)、[Swift CLI 启动逻辑](https://github.com/modelscope/ms-swift/blob/main/swift/cli/main.py)。当前实现使用 `--tuner_type` 和 `--infer_backend transformers`，面向 4.x；不要混用旧版本的 `--train_type`、`--infer_backend pt`。
