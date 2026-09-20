# 筛选实验

目前仍在验证方案，各脚本保留独立入口。

## parent_child.py：当前实验

```bash
python3 experiments/parent_child.py examples/synthetic-transcript.json --out runs/preview
```

省略 `--run` 只准备本地请求；显式加 `--run` 才发送给 TypeSafe，需要 `TYPESAFE_API_KEY`。每次使用新输出目录。

固定参数为 60 秒父块、前后各 15 秒上下文、每片一个纯闲聊判断和 0.90 删除阈值。片段 ID、原文、时间戳不改写。同一父块的独立问题合并到一次 HTTP 请求。

输入使用 `meeting_notes.py process` 导出的规范化 JSON，或仓库合成样例。子片段需包含唯一 id、start、end、text、source。

## 第一轮基线

`meeting_notes.py prepare ... --filter jev` 使用原来的五类判断和约一分钟整块筛选，用于历史比较，不是当前单问题实验入口。

## 早期合成实验

- `python3 tools/jev_smoke_test.py --dry-run`：准备三个合成请求；去掉 `--dry-run` 会调用 API。
- `python3 experiments/synthetic_baseline.py`：20 个合成发言、100 个判断，会直接调用 API，结果写到 `runs/synthetic-baseline/`。

更改参数时记录问题、模型版本、阈值和分段，保留原始概率及候选删除项。不要用字符减少率代替 token 统计，也不要改阈值后只展示更好看的结果。
