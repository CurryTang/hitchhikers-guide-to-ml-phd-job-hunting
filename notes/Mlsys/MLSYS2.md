# MLSYS2 · CUDA 编程模型与 GPU 组件

## More GPU

### Overview

![[assets/Pasted image 20251222135239.png]]
**图1：NVIDIA H100/B100 GPU 整体架构抽象图。** 展示了 GPU 的层次化内存与计算结构：多个流多处理器（SM 0, SM 1, ... SM N-1）并行排列，每个 SM 内包含 4 个 Tensor Core（负责矩阵乘法运算，贡献主要算力，类似 TPU 的 MXU）和 4 个 Warp Scheduler（SIMD 向量单元，包含 32 个 lane 即"CUDA Core"，同一 warp 内所有 lane 必须执行相同操作）。每个 SM 拥有 256KB 的 L1 Cache/SMEM（共享内存，可由程序员控制，类似 TPU VMEM 但更小）。所有 SM 共享 50MB 的 L2 Cache（硬件自动管理，提供更快的带宽）和底层的 HBM 高带宽显存（H100 为 80GB，B100 为 192GB，用于存储模型参数、激活值和优化器状态）。

![[assets/Pasted image 20251222135741.png]]
**图2：NVIDIA H100 单个 SM（流多处理器）内部详细架构图。** 每个 SM 包含 4 个处理块（Processing Block），共享 L1 指令缓存和 256KB L1 数据缓存/共享内存。每个处理块包含：L0 指令缓存、Warp Scheduler（每周期调度 32 线程）、Dispatch Unit、16384×32-bit 的寄存器文件，以及大量计算单元——16 个 INT32 单元、16 个 FP32 单元、8 个 FP64 单元、1 个第四代 Tensor Core、LD/ST（加载/存储）单元和 SFU（特殊函数单元）。底部还配备 Tensor Memory Accelerator（张量内存加速器）和 Tex（纹理单元）。这种设计使得 H100 能够高效地并行执行大规模矩阵运算和深度学习工作负载。


### 组件

#### GPU 计算组件总结

| 层级                 | 组件                            | 数量 (H100)     | 角色      | 负责的操作                            |
| ------------------ | ----------------------------- | ------------- | ------- | -------------------------------- |
| **GPU 级**          | GigaThread Engine             | 1             | 全局调度器   | 将 thread block 分配到各个 SM          |
| **SM 级**           | SM (Streaming Multiprocessor) | 132           | 独立计算单元  | 执行一个或多个 thread block，管理内部资源      |
| **SubPartition 级** | Warp Scheduler                | 4 per SM      | Warp 调度 | 从 warp pool 中选择 eligible warp 发射 |
|                    | Dispatch Unit                 | 2 per SubPart | 指令分发    | 读操作数、选执行单元、发射指令                  |
|                    | Scoreboard                    | 1 per SubPart | 依赖追踪    | 跟踪寄存器状态，检测数据冒险                   |
| **执行单元级**          | Tensor Core                   | 4 per SM      | 矩阵乘法    | GEMM，~1024 FLOPs/cycle，占 93%+ 算力 |
|                    | FP32 CUDA Cores               | 128 per SM    | 单精度浮点   | ReLU、pointwise ops、reduction     |
|                    | FP64 CUDA Cores               | 64 per SM     | 双精度浮点   | 科学计算（ML 中很少用）                    |
|                    | INT32 Cores                   | 64 per SM     | 整数运算    | 地址计算、索引、位操作                      |
|                    | Load/Store Units              | 32 per SM     | 内存访问    | 发起 load/store 请求，地址计算            |
|                    | SFU (Special Function Unit)   | 16 per SM     | 特殊函数    | sin, cos, exp, rsqrt 等超越函数       |
|                    | Texture Units                 | 4 per SM      | 纹理采样    | 图形渲染用，ML 中偶尔用于插值                 |

#### 计算组件层级关系

```
GPU
 └── GigaThread Engine (全局调度)
      └── SM ×132
           ├── Warp Pool (最多 64 warps 常驻)
           └── SubPartition ×4
                ├── Warp Scheduler ──► 选 warp
                ├── Dispatch Unit ×2 ──► 发指令
                └── Execution Units
                     ├── Tensor Core (矩阵乘)
                     ├── FP32 Cores ×32 (向量算术)
                     ├── INT32 Cores ×16
                     ├── FP64 Cores ×16
                     ├── LD/ST Units ×8
                     └── SFU ×4
```

#### GPU 存储组件总结

| 层级 | 组件 | 容量 (H100) | 带宽 | 延迟 | 作用域 | 用途 |
|------|------|-------------|------|------|--------|------|
| **片外** | HBM (显存) | 80 GB | 3.35 TB/s | ~400 cycles | 全局 | 模型权重、激活、大 tensor |
| | L2 Cache | 50 MB | ~12 TB/s | ~100 cycles | 全局 | 自动缓存 HBM 数据 |
| **SM 级** | SMEM (Shared Memory) | 256 KB per SM | ~33 TB/s | ~20 cycles | Block 内共享 | Tile 数据、线程间通信 |
| | L1 Cache | 与 SMEM 共享 | ~33 TB/s | ~20 cycles | SM 私有 | 自动缓存（可配置比例） |
| | TMEM (Tensor Memory) | B200 新增 | 极高 | 极低 | SubPart 私有 | 喂 Tensor Core 的专用缓存 |
| **线程级** | Register File | 64K ×32bit per SM | ~80 TB/s | 1 cycle | 线程私有 | 局部变量、中间结果 |
| | Local Memory | 溢出到 HBM | 同 HBM | 高 | 线程私有 | 寄存器溢出 (register spill) |
| **特殊** | Constant Memory | 64 KB | 广播优化 | ~4 cycles (cached) | 只读全局 | 常量参数、超参数 |
| | Texture Memory | 与 L1 共享 | 空间局部性优化 | 中等 | 只读全局 | 2D 空间数据访问 |

#### 存储层级金字塔

```
                    ┌─────────┐
                    │ Register│  64K×32bit/SM, 1 cycle, ~80 TB/s
                    │  File   │  线程私有
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  SMEM   │  256 KB/SM, ~20 cycles, ~33 TB/s
                    │L1 Cache │  Block 共享 / 自动缓存
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │L2 Cache │  50 MB, ~100 cycles, ~12 TB/s
                    │         │  全局共享，自动管理
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │   HBM   │  80 GB, ~400 cycles, 3.35 TB/s
                    │ (DRAM)  │  全局，持久存储
                    └─────────┘

容量:    小 ◄─────────────────────────────► 大
速度:    快 ◄─────────────────────────────► 慢
```

#### 各存储的典型使用场景

| 存储 | ML 中的典型用途 | 编程方式 |
|------|----------------|---------|
| **Register** | 累加器、循环变量、Tensor Core 输入输出 | 自动分配，局部变量 |
| **SMEM** | GEMM tiling、attention 的 K/V cache、reduction 中间结果 | `__shared__` 显式声明 |
| **L2** | 跨 SM 复用的数据（如同一 batch 的不同 head） | 自动，可用 `cudaAccessPolicyWindow` 提示 |
| **HBM** | 权重矩阵、输入输出 tensor、optimizer state | `cudaMalloc`，全局数组 |
| **Constant** | Layer 的超参数、lookup table | `__constant__` 声明 |


### 通过伪代码理解warp和dispatch的工作机制


```python
# ============ SM 内部结构 ============
class SM:
    def __init__(self):
        # 执行单元（以 Ampere 架构为例，每个 SM 有 4 个 sub-partition）
        self.sub_partitions = [SubPartition() for _ in range(4)]
        
        # 每个 sub-partition 有自己的 warp scheduler + dispatch unit
        
class SubPartition:
    def __init__(self):
        self.warp_scheduler = WarpScheduler()
        self.dispatch_units = [DispatchUnit(), DispatchUnit()]  # 通常 2 个
        
        # 执行单元
        self.int32_units = [INT32_ALU() for _ in range(16)]
        self.fp32_units = [FP32_ALU() for _ in range(16)]
        self.fp64_units = [FP64_ALU() for _ in range(8)]
        self.ld_st_units = [LoadStoreUnit() for _ in range(8)]
        self.sfu_units = [SpecialFuncUnit() for _ in range(4)]  # sin, cos, exp...
        self.tensor_cores = [TensorCore() for _ in range(1)]


# ============ Warp Scheduler ============
class WarpScheduler:
    """决定下一个周期执行哪个 warp"""
    
    def __init__(self):
        self.warp_pool = []  # 该 scheduler 管理的所有 warp（通常 8 个左右）
        
    def select_warps_to_issue(self):
        """每个周期选择可以发射的 warp"""
        
        ready_warps = []
        for warp in self.warp_pool:
            if self.is_warp_eligible(warp):
                ready_warps.append(warp)
        
        # 调度策略：GTO (Greedy Then Oldest), LRR (Loose Round Robin), 等
        selected = self.scheduling_policy(ready_warps)
        return selected  # 可能返回 1-2 个 warp（取决于 dispatch unit 数量）
    
    def is_warp_eligible(self, warp):
        """检查 warp 是否可以被调度"""
        
        if warp.is_finished():
            return False
            
        # 检查 scoreboard：指令的操作数是否就绪
        next_inst = warp.get_next_instruction()
        if not self.scoreboard.operands_ready(warp.id, next_inst):
            return False  # 数据依赖，stall
            
        # 检查结构冒险：目标执行单元是否可用
        if not self.check_structural_hazard(next_inst):
            return False
            
        # 检查是否在等待 barrier 同步
        if warp.waiting_at_barrier:
            return False
            
        return True
    
    def scheduling_policy(self, ready_warps):
        """调度策略示例：GTO - 优先让同一个 warp 连续执行"""
        if not ready_warps:
            return []
        
        # 优先选上次执行的 warp（局部性）
        if self.last_issued_warp in ready_warps:
            return [self.last_issued_warp]
        
        # 否则选最老的 ready warp
        return [min(ready_warps, key=lambda w: w.age)]


# ============ Dispatch Unit ============
class DispatchUnit:
    """把 warp scheduler 选中的指令分发到执行单元"""
    
    def dispatch(self, warp, instruction):
        """将指令分发到具体执行单元"""
        
        # 1. 从 Register File 读取操作数（32 个线程的数据）
        operands = self.read_operands(warp, instruction)
        # operands 是 32 份数据，每个 lane 一份
        
        # 2. 根据指令类型选择执行单元
        exec_unit = self.select_execution_unit(instruction)
        
        # 3. 发射到执行单元
        exec_unit.issue(warp.id, warp.active_mask, instruction, operands)
        
        # 4. 更新 scoreboard：标记目标寄存器为 pending
        self.scoreboard.mark_pending(warp.id, instruction.dest_reg)
        
    def select_execution_unit(self, instruction):
        match instruction.opcode:
            case 'FADD' | 'FMUL' | 'FFMA':
                return self.find_available(self.fp32_units)
            case 'IADD' | 'IMUL' | 'IMAD':
                return self.find_available(self.int32_units)
            case 'LD' | 'ST':
                return self.find_available(self.ld_st_units)
            case 'SIN' | 'COS' | 'EXP' | 'RCP':
                return self.find_available(self.sfu_units)
            case 'HMMA' | 'IMMA':  # Tensor Core ops
                return self.find_available(self.tensor_cores)


# ============ 执行单元 ============
class FP32_ALU:
    """FP32 执行单元 - SIMT 执行"""
    
    def issue(self, warp_id, active_mask, instruction, operands):
        """执行 32 个线程的计算"""
        
        results = [None] * 32
        for lane in range(32):
            if active_mask & (1 << lane):  # 只执行 active 的线程
                a = operands.src1[lane]
                b = operands.src2[lane]
                
                match instruction.opcode:
                    case 'FADD':
                        results[lane] = a + b
                    case 'FMUL':
                        results[lane] = a * b
                    case 'FFMA':
                        c = operands.src3[lane]
                        results[lane] = a * b + c
        
        # 写回 register file（流水线化，可能需要几个周期）
        self.writeback_queue.enqueue(warp_id, instruction.dest_reg, results)


# ============ 完整的每周期流程 ============
def sm_cycle(sub_partition):
    """每个时钟周期的流水线操作"""
    
    # Stage 1: Warp Scheduler 选择 warp
    selected_warps = sub_partition.warp_scheduler.select_warps_to_issue()
    
    # Stage 2: Dispatch Unit 分发指令
    for i, warp in enumerate(selected_warps):
        if i < len(sub_partition.dispatch_units):
            instruction = warp.fetch_next_instruction()
            sub_partition.dispatch_units[i].dispatch(warp, instruction)
            warp.pc += 1
    
    # Stage 3-N: 执行单元流水线执行（并行进行）
    for unit in all_execution_units(sub_partition):
        unit.pipeline_tick()
    
    # Writeback: 完成的结果写回 register file，更新 scoreboard
    sub_partition.process_writebacks()
```

## 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                         SM                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Sub-Partition (×4)                         │ │
│  │                                                         │ │
│  │   ┌──────────────┐                                      │ │
│  │   │ Warp Pool    │  (8 warps)                          │ │
│  │   │ W0 W1 W2 ... │                                      │ │
│  │   └──────┬───────┘                                      │ │
│  │          │ 哪个 warp ready?                              │ │
│  │          ▼                                              │ │
│  │   ┌──────────────┐                                      │ │
│  │   │Warp Scheduler│ ──选择 1-2 个 eligible warp          │ │
│  │   └──────┬───────┘                                      │ │
│  │          │                                              │ │
│  │          ▼                                              │ │
│  │   ┌──────────────┐    ┌──────────────┐                  │ │
│  │   │Dispatch Unit │    │Dispatch Unit │  (×2)            │ │
│  │   └──────┬───────┘    └──────┬───────┘                  │ │
│  │          │                   │                          │ │
│  │          ▼                   ▼                          │ │
│  │   ┌─────────────────────────────────────────────┐       │ │
│  │   │           Execution Units                    │       │ │
│  │   │  INT32  FP32  FP64  LD/ST  SFU  TensorCore  │       │ │
│  │   └─────────────────────────────────────────────┘       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

| 组件                 | 职责                     | 类比             |
| ------------------ | ---------------------- | -------------- |
| **Warp Scheduler** | 决定"谁来执行"，检查依赖、冒险、选择策略  | 调度员：选择下一个上场的选手 |
| **Dispatch Unit**  | 决定"怎么执行"，读操作数、选执行单元、发射 | 分配员：把选手送到正确的赛道 |

Cycle 1:  Warp_A: LD r1, [addr]     # 发起内存读取，需要等待 ~400 周期
Cycle 2:  Warp_B: ADD r2, r3, r4    # 切换到 B
Cycle 3:  Warp_C: MUL r5, r6, r7    # 切换到 C
...
Cycle 400: Warp_A: (内存返回)       # A 的数据到了
Cycle 401: Warp_A: ADD r8, r1, r9   # A 继续执行

### Tensor Core 的重要性

下面的数字用 H100 SXM 的量级做直觉对比：dense BF16 Tensor Core 约 990 TFLOPs，FP32 CUDA Core 约 60-66 TFLOPs。比例不是为了精确 benchmark，而是说明现代 ML workload 的主计算路径在哪里。

<div class="compute-share-card">
  <div class="compute-share-head">
    <span>H100 compute path</span>
    <strong>GEMM dominates ML throughput</strong>
  </div>
  <div class="compute-share-row tensor">
    <div class="compute-share-label">
      <strong>Tensor Core</strong>
      <span>BF16 dense GEMM</span>
    </div>
    <div class="compute-share-track"><span class="share-94"></span></div>
    <div class="compute-share-value">~990 TFLOPs · ~94%</div>
  </div>
  <div class="compute-share-row cuda">
    <div class="compute-share-label">
      <strong>CUDA Core</strong>
      <span>FP32 scalar/vector path</span>
    </div>
    <div class="compute-share-track"><span class="share-6"></span></div>
    <div class="compute-share-value">~60-66 TFLOPs · ~6%</div>
  </div>
  <div class="compute-share-foot">
    2:4 structured sparsity can roughly double Tensor Core peak, so the gap can be even larger for sparse-friendly GEMM.
  </div>
</div>

结论要更精确地说：Transformer 里的 projection、MLP、attention GEMM 需要尽量落到 Tensor Core；CUDA Core 仍然负责地址计算、分支、elementwise、reduction、softmax、normalization 和各种 glue code。优化 compute-bound kernel 时，如果主路径没有用上 Tensor Core，通常先不要谈更细的调度技巧。


Ref: 
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.

---

## 课后练习题

### 练习 1：Tensor Core vs CUDA Core

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">为什么 GEMM 优化首先关心 Tensor Core？</span></summary>

现代训练和推理里的大头是矩阵乘：attention projection、MLP、MoE expert、QK 和 PV 都是 GEMM-like workload。H100 上 BF16 Tensor Core peak 远高于 FP32 CUDA Core peak。如果主计算没有落到 Tensor Core，通常说明数据类型、矩阵形状、layout、alignment 或 kernel lowering 有问题。

</details>

### 练习 2：SM 内部瓶颈定位

<details class="exercise">
<summary><span class="q-label">答案</span> <span class="q-text">occupancy 很高但 Tensor Core 利用率低，可能是什么原因？</span></summary>

occupancy 高只表示有足够 warps 常驻，不保证发射的是高吞吐指令。Tensor Core 利用率低可能来自没有使用 mma/wgmma 路径、tile shape 不匹配、global/shared memory 供数不足、register spill、同步过多，或者 kernel 主体其实是 softmax/reduction 这类非 GEMM 操作。

</details>
