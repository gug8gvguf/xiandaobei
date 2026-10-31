# FP32基准运行说明

## 1. 哪个文件是统一基准

`baseline/solution.py` 是本题已经实现、可以运行的原始FP32基准。它以题目包内的ProGen2-medium公共权重和统一Tokenizer为输入，使用官方模型计算实现、动态KV Cache和确定性生成流程。

基准的作用有三个：确认环境与模型能正常运行；帮助理解模型如何加载和生成；作为报告中“优化前”的统一对照。它不是性能优化答案。不要改写原始基准后仍把它称为统一基准。

`submission/solution.py` 是学生待实现骨架，不会自动加载或转发到基准。学生需要完成自己的模型加载和生成实现。可以阅读基准理解调用方式，具体优化需要自行设计、实现和验证。

## 2. 第一次运行

以下命令都在题目包根目录 `progen2_contest/` 执行：

```bash
python check_environment.py --full-hash
bash run_baseline.sh
```

`run_baseline.sh` 不接收目录参数，固定测试 `baseline/`。脚本先检查环境，再自动运行单条生成、多条生成、生成较长、输入较长四项规定测试任务。模型只加载一次，每项用另一组同规格输入预热1次，再对规定测试输入计时生成1次。

`ENVIRONMENT_CHECK_PASSED` 仅表示环境与公共文件通过基础检查，不代表模型权重已成功加载。请继续运行基准，以四项测试完成为准；若模型加载报错，应核对镜像内 PyTorch、Transformers 等版本，不要通过修改参考答案或测试程序处理。

结果写入 `results/baseline_public.json`。再次运行会覆盖这个文件，需要保留时先另存。

测试完成时应出现：

```text
PUBLIC_INTERFACE_TEST_PASSED (...)
```

基准在已验证环境中与随包FP32参考结果逐字一致。若自己的环境出现差异，应保留完整结果和版本信息核对，不要通过修改参考答案消除差异。

## 3. 生成规则是什么

ProGen2先读取输入前缀，再逐步预测新Token。本题每一步从20种大写标准氨基酸对应的Token中选择得分最高者，得分相同时按下面的固定字母顺序选择第一个：

```text
ACDEFGHIKLMNPQRSTVWY
```

关闭随机采样，固定生成规定数量的字母；不使用temperature、top-k或top-p改变生成过程，也不因EOS提前停止。输入开头的 `1` 或 `2` 是ProGen控制标记，需要保留；它们不计入新生成字母，也不能出现在返回后缀中。

公共权重和Tokenizer相同、生成规则相同，才能比较优化前后的结果。此处的字母一致率反映与FP32基准计算结果的接近程度，不等于蛋白质生物学功能得分。

## 4. 如何测试自己的程序

完成 `submission/solution.py` 后运行：

```bash
bash run_public_test.sh submission
```

省略目录参数也默认测试 `submission/`。需要测试其他目录时，填写该目录路径：

```bash
bash run_public_test.sh my_submission
```

默认结果为 `results/submission_public.json`。两个Bash入口只接受说明中的参数，不要向它们添加不存在的 `--repeat`、`--warmup` 等选项。

需要为多个方案保存独立结果时，可以直接使用测试器的 `--output`：

```bash
python public_test.py --submission baseline --output results/baseline_v1.json
python public_test.py --submission submission --output results/submission_v1.json
```

如果作品包含 `build.sh`，先使用 `bash run_public_test.sh submission` 完成环境检查和构建，再直接调用Python测试器。直接调用 `public_test.py` 不代替Bash入口的构建步骤。

## 5. 结果怎么看

| 字段或提示 | 含义 |
|---|---|
| `time_s` | 该负载一次完整生成调用及DCU同步的耗时，单位秒 |
| `same` | 与统一FP32参考在相同位置上相同的字母数量 |
| `total` | 该负载规定的新生成比较位置数，即B×G |
| `consistency_rate` | `same/total`，例如1.0表示100% |
| `exact_match` | 是否所有规定位置都完全相同 |
| `PUBLIC_INTERFACE_TEST_PASSED` | 四种负载运行完成且输出格式合格，不代表全部字母一致，也不是最终成绩 |

比较基准与优化程序时，对应负载必须一致。报告中的整模型加速比可按以下方式计算：

```text
整模型加速比 = 同负载FP32基准耗时 ÷ 同负载优化程序耗时
```

例如相同负载下基准10秒、优化8秒，加速比为1.25倍。这里的基准加速比用于说明优化效果；性能分则比较所有合格队伍在该任务上的最短时间，每项最多10分，不能用本队自行准备的慢版本或FP32基准时间代替。

作品测试即使结果不完全相同，也可能提示接口通过；涉及精度变化的方案尤其要查看实际 `consistency_rate`，不能只看最后一行。

## 6. 测试记录与成绩计算

各队使用 `public_test_data/public_cases.json` 中的固定输入和FP32参考结果。每种负载使用另一组同规格输入预热1次，再计时运行1次。

一次计时覆盖Tokenizer处理、输入构造、传输、Prefill、Decode和输出字符串转换，以及必要DCU同步；模型加载、离线编译、统一预热、结果校验和写盘不计时。预热后可以保留模型和输入无关的通用资源，但不能沿用预热产生的生成结果或输入相关KV Cache。

完整日志是作品必交材料。请在题目包根目录使用以下命令，分别代替前面的基准和作品运行命令；无需先运行一遍再为保存日志重复运行：

```bash
mkdir -p results
set -o pipefail
bash run_baseline.sh 2>&1 | tee results/baseline.log
bash run_public_test.sh submission 2>&1 | tee results/submission.log
```

上述命令将完整终端输出分别保存为 `results/baseline.log` 和 `results/submission.log`；测试正常完成时，仍生成原有的JSON结果。两个日志文件须连同 `baseline_public.json`、`submission_public.json` 一起放入 `submission/evidence/` 提交，复制方法见 [提交要求](SUBMISSION.md)。日志应保留环境信息、四项任务结果及警告或报错，不能用截图或手工摘录代替。

`set -o pipefail` 用于保留测试命令的失败状态。测试开发过程中可以重复运行；最终须提交与交付代码版本对应的一轮完整作品测试记录及日志，以及同规格FP32基准记录及日志，不要求提交全部调试历史。不得按负载挑选不同轮次或不同代码版本的最好结果。再次使用上述命令会覆盖同名日志，需要保留的其他实验记录应先另存。

测试器遇到异常会直接退出，可能不生成本轮JSON；此时应保留本轮完整日志并说明原因。已完成且可核验的负载可以据日志计分，失败、未执行或无有效记录的负载一致率和性能分为0；基础运行分为0，其他有效负载和报告仍独立计分。不得用旧JSON替代本轮记录。

正常完成以JSON原始数值为准；异常时仅有可核验日志的，按日志记录值核算。

学生不需要编写评分程序，也不需要手动切换四种负载。详细评分规则见 [赛题说明](赛题说明.docx)。

