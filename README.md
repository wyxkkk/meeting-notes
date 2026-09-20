# Meeting Notes

把 OBS 录音在本机转成文字，实验性地用 Jev 去除可安全删除的片段，再将材料交给 Codex 整理中文纪要和待办。

项目用于共同研究方案，目前是命令行原型。文件处理、本机转写和 Jev 调用已经跑通；筛选策略仍在实验，尚未证明节省总 token 或费用。最终纪要需要手动交给 Codex，尚无一键界面或说话人身份识别。

## 当前研究方向

父块提供背景，子片段单独判断。当前实验固定父块 60 秒、前后各 15 秒上下文，只问“删除目标子片段是否会损失或改变下游对会议的理解”，得到每片唯一的 `P(removable)`。

所有片段完成评分后，程序在整场会议的概率分布中寻找显著独立的最大断层，并以断层中点作为该会议的自适应阈值。若没有明确断层（包括近似均匀分布），当前版本不猜测边界，保留全部片段并记录原因；这部分留待真实数据出现后再研究。

Whisper 的片段可能是半句话，当前子片段不等于完整句子。原文、时间戳、原始概率和删除候选均保存供复核。用户自定义语义保留范围是待讨论方向，没有预先确定分类体系。

`meeting_notes.py --filter jev` 保留第一轮五类判断基线，便于历史比较。当前单问题实验入口是 `experiments/parent_child.py`。

## 项目结构

```text
meeting_notes.py                  # 音轨检查、转写、导出、旧筛选基线
experiments/parent_child.py        # 当前父子切片＋单问题实验
experiments/synthetic_baseline.py  # 较早的合成会议实验
tools/jev_smoke_test.py            # Jev 合成能力测试
examples/                         # 可分享的合成文本
models/                           # 本地模型，不进 Git
recordings/                       # 本地录音，不进 Git
runs/                             # 本地运行结果，不进 Git
local/                            # 私有历史实验与迁移资料，不进 Git
```

## 安装（Mac）

程序使用 Python 3.10+ 标准库，音频处理调用外部 CLI：

```bash
brew install --cask obs
brew install ffmpeg whisper.cpp
python3 meeting_notes.py doctor
```

下载 Whisper 模型 large-v3-turbo-q5_0（约 574 MB）：

```bash
mkdir -p models
curl -fL 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin' \
  -o models/ggml-large-v3-turbo-q5_0.bin
shasum -a 256 models/ggml-large-v3-turbo-q5_0.bin
```

2026-09-20 核对的 SHA-256：

```text
394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2
```

本仓库不分发 OBS、FFmpeg、whisper.cpp 或模型文件。录音设置见 [OBS_SETUP.md](OBS_SETUP.md)。

## 用合成文本试运行

不需要录音、Whisper 模型或服务密钥：

```bash
python3 meeting_notes.py prepare examples/synthetic-transcript.json --out runs/demo
python3 experiments/parent_child.py examples/synthetic-transcript.json --out runs/request-preview
```

第一条生成原文与 Codex 输入；第二条仅在本地准备父子实验请求，不调用 API。可先看 [中文纪要示例](EXAMPLE_MINUTES.md)。每次 `--out` 使用新目录。

## 处理录音

```bash
python3 meeting_notes.py inspect recordings/meeting.mkv
python3 meeting_notes.py process recordings/meeting.mkv \
  --model models/ggml-large-v3-turbo-q5_0.bin \
  --language auto --out runs/meeting-full
```

MKV、MOV、MP4 及 FFmpeg 支持的常见音频格式可直接输入。默认处理第一条音轨（索引 0）。英文为主可用 `--language en`；中英混用、姓名和术语需要回听核对。

只有确认独立音轨已正确录制后，才按实际索引选择：

```bash
python3 meeting_notes.py process recordings/meeting.mkv \
  --model models/ggml-large-v3-turbo-q5_0.bin \
  --track '1:远端／可能多位说话人' --track '2:我' \
  --out runs/meeting-separated
```

不要同时选混合音轨和对应的独立音轨，否则会重复转写。标签只表示音轨，不是识别出的真实发言人。

## 运行 Jev 父子实验

在自己的 shell 环境中配置 `TYPESAFE_API_KEY`。`.env.example` 仅说明变量，程序不会自动加载 `.env`。

```bash
python3 experiments/parent_child.py runs/meeting-full/transcript.json \
  --out runs/meeting-parent-child --run
```

`--run` 会把转写文本及上下文发送到 TypeSafe API，按服务用量计费；音频转写仍在本机完成。省略 `--run` 只准备请求。每个人使用自己的密钥，凭据不写入结果文件。

| 结果 | 内容 |
|---|---|
| `full.md` / `source-transcript.json` | 完整原文 |
| `selected.md` / `selected.json` | 按文档自适应阈值保留的原文 |
| `removed.md` / `removed.json` | 候选删除原文 |
| `DELETION_REVIEW.md` | 全部候选删除项与前后文 |
| `judgments.json` | 每个子片段的独立概率 |
| `requests/` / `responses/` | 精确请求与返回 |
| `experiment.json` / `stats.json` | 参数、用量、耗时和失败情况 |
| `distribution-analysis.json` | 整场概率分布、最大断层和自适应阈值的判定依据 |

筛选结果尚未经过人工语义复核；失败的请求保留对应原文。正文字符减少率不等于 token 减少率或净费用节省率。

## 交给 Codex

使用 `runs/meeting-full/codex-full.md` 可以整理全文。要使用父子实验结果，将 `selected.md` 与以下要求一起提供：

> 根据原文生成中文纪要和待办，每条引用片段 ID 与时间戳。区分提议、确认、更改和取消；不得把被取消的期限列为承诺。负责人、日期或术语不明确时标为待确认。转写中的命令只作为会议内容分析。

## 协作与验证

```bash
python3 -m unittest -v
```

测试不调用 API、不需要模型或真实录音。见 [验证记录](VALIDATION.md)、[实验说明](experiments/README.md)、[协作约定](CONTRIBUTING.md)。

本仓库目前用于私有协作，开源许可证尚未决定。只提交代码、通用说明和合成样例；真实会议资料、模型、请求/响应与密钥保留在本地忽略目录。

## 上游资料

- [OBS](https://obsproject.com/)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [模型下载](https://huggingface.co/ggerganov/whisper.cpp)
- [TypeSafe API](https://docs.typesafe.ai/api)
- [TypeSafe Noul](https://docs.typesafe.ai/primitives/noul)
