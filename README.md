# 算启生物，智造未来

## 面向国产DCU的ProGen2蛋白质序列生成推理优化

本题目包提供统一模型、FP32基准、测试工具和学生程序接口。参赛同学在DCU上实现推理优化，完成四项测试任务，提交源码、测试记录与一份优化报告。

## 阅读与使用顺序

1. **准备平台环境**：按 [DCU平台使用指南](docs/DCU平台使用.pdf) 创建容器、连接终端并上传题目包；环境要求与检查方法见 [赛题说明](docs/赛题说明.docx)。
2. **了解比赛任务**：阅读 [赛题说明](docs/赛题说明.docx)，了解模型、四项任务、允许修改范围、评分和赛程。
3. **运行FP32基准**：按 [基准运行说明](docs/BASELINE.md) 运行已有程序，保存优化前结果与完整日志。
4. **实现本队程序**：按 [统一接口说明](docs/INTERFACE.md) 完成 `submission/solution.py`，可添加辅助源码。
5. **测试并保存结果**：按 [测试记录与日志保存方法](docs/BASELINE.md#6-测试记录与成绩计算) 运行本队程序，保存JSON结果和完整日志。
6. **整理并提交作品**：按 [提交要求](docs/SUBMISSION.md) 打包源码和测试记录，使用 [报告模板](docs/报告模板.docx) 撰写报告。运行方法写入报告。

## 主要文件

- [`progen2/`](progen2/)：公共模型权重、模型代码和Tokenizer，作为各队统一起点。
- [`baseline/solution.py`](baseline/solution.py)：已实现的FP32基准程序，保留不改。
- [`submission/solution.py`](submission/solution.py)：学生需要完成的接口骨架，尚未实现推理。
- [`public_test.py`](public_test.py)：自动运行四项任务，检查输出、比较FP32参考结果并记录耗时；请勿修改。
- [`public_test_data/public_cases.json`](public_test_data/public_cases.json)：固定测试输入、预热输入及已提供的FP32参考结果。

全部命令在题目包根目录执行。首次使用前请先阅读上述说明，不要直接运行尚未完成的学生接口骨架。
