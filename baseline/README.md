# 统一FP32原始基准

`solution.py`是提供的可运行FP32动态KV参考实现，保持原算法未做优化组合。它用于确认公共模型、输入输出和基准耗时。

从题目包根目录运行：

```bash
bash run_baseline.sh
```

程序自动测试四种规定负载，结果写入`results/baseline_public.json`。完整说明见`docs/BASELINE.md`。

请保留此目录原样作为统一对照，在`submission/`目录实现自己的程序。
