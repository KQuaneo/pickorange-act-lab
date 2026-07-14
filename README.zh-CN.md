# PickOrange-ACT：可审计的长时序具身智能实验

[English](README.md) · [完整实验报告](docs/EXPERIMENT_REPORT.md) · [复现说明](docs/REPRODUCIBILITY.md) · [机器可读结果](results/summary.json)

这是一个基于 LeIsaac、Isaac Lab 和 LeRobot ACT 的 SO-101 三橘子长时序
抓取放置项目。项目覆盖专家数据采集与切片、严格前缀审计、ACT 训练、GPU
容错流水线、协议安全测评和失败归因，而不是只展示一次成功视频。

## 最重要的结果

- 30 条专家数据；最终每个配置评测 20 episodes，seed=2026。
- 单策略 A0 在 21k/30k/36k/42k 均为 0/20。
- 三子策略固定时间调度 A1 在 10k 达到 2/20，在 14k 达到 **3/20=15%**。
- isolated B1/B2/B3 在 14k 分别为 30%/45%/30%；B2/B3 是 oracle 初始化，
  只能视为理想前置状态下的 primitive 能力上限。
- B3 数据审计为 integrity 30/30、target success 29/30、strict-prefix
  success 28/30，排除了两条语义不合格数据。
- 审计发现第三次释放发生在 350–358 actions，证明旧的 340-action 切片会
  截断关键动作，因此正式 A1 改为每阶段 420 actions。

![最终完整任务结果](assets/final-full-task-results.svg)

## 项目体现的能力

1. **研究设计**：A0/A1/A2/A3、isolated primitives、SingleOrange horizon
   sweep 和 checkpoint 对比，区分规划、低层控制与阶段衔接问题。
2. **数据工程**：按稳定抓取/放置事件切片，检查目标成功与所有前缀橘子是否
   仍在盘中，而不是把固定帧数当作成功标签。
3. **评测工程**：显式区分 `native_horizon` 与可选 `matched_horizon`，记录
   action 数、simulation steps、理论时间和初始化来源，避免静默覆盖历史结果。
4. **诊断能力**：记录 fixed-time scheduler 的 post-success overrun、阶段
   start-state deviation、prefix 破坏与失败原因。
5. **系统可靠性**：所有长流程使用 tmux/supervisor，支持完成标志、断点复用、
   OOM 后降低并行度、指数退避、磁盘保护和自动衔接下一阶段。
6. **科研诚信**：保留 0/20 和宽置信区间，不把 isolated oracle 结果包装成
   端到端成功，也不报告已取消的 50-demo 实验。

## 结论

最终结果说明：任务分解确实产生了 A0 未观察到的完整成功，但 15% 仍不足以
证明系统稳定。低层接触鲁棒性、固定时间切换造成的阶段尾部空转，以及前一阶段
误差导致的 start-state 分布漂移都值得继续研究。当前数据只支持相关性和工程
诊断，不支持强因果结论。

代码入口和运行方式见 [复现说明](docs/REPRODUCIBILITY.md)，完整实验演进与限制见
[统一报告](docs/EXPERIMENT_REPORT.md)。

