# MLSYS2 · CUDA Programming Model and GPU Components

## More GPU

### Overview

![[assets/Pasted image 20251222135239.png]]
**Figure 1: Abstract diagram of the overall NVIDIA H100/B100 GPU architecture.** It shows the GPU's hierarchical memory and compute structure: multiple streaming multiprocessors (SM 0, SM 1, ... SM N-1) are arranged in parallel, and each SM contains 4 Tensor Cores (responsible for matrix multiplication and contributing most of the compute throughput, analogous to the TPU MXU) and 4 Warp Schedulers (SIMD vector units containing 32 lanes, i.e., "CUDA Cores"; all lanes within the same warp must execute the same operation). Each SM has 256KB of L1 Cache/SMEM (shared memory that can be controlled by the programmer, similar to TPU VMEM but smaller). All SMs share a 50MB L2 Cache (automatically managed by hardware to provide faster bandwidth) and the underlying HBM high-bandwidth memory (80GB on H100, 192GB on B100), which stores model parameters, activations, and optimizer states.

![[assets/Pasted image 20251222135741.png]]
**Figure 2: Detailed internal architecture of a single NVIDIA H100 SM (streaming multiprocessor).** Each SM contains 4 processing blocks that share an L1 instruction cache and a 256KB L1 data cache/shared memory. Each processing block contains an L0 instruction cache, a Warp Scheduler (scheduling 32 threads per cycle), a Dispatch Unit, a 16384×32-bit register file, and many compute units—16 INT32 units, 16 FP32 units, 8 FP64 units, 1 fourth-generation Tensor Core, LD/ST (load/store) units, and an SFU (special function unit). The bottom also includes a Tensor Memory Accelerator and Tex (texture units). This design enables the H100 to efficiently execute large-scale matrix operations and deep learning workloads in parallel.


### Components

#### Summary of GPU compute components

| Level | Component | Count (H100) | Role | Operations handled |
| ------------------ | ----------------------------- | ------------- | ------- | -------------------------------- |
| **GPU level** | GigaThread Engine | 1 | Global scheduler | Assigns thread blocks to SMs |
| **SM level** | SM (Streaming Multiprocessor) | 132 | Independent compute unit | Executes one or more thread blocks and manages internal resources |
| **SubPartition level** | Warp Scheduler | 4 per SM | Warp scheduling | Selects eligible warps from the warp pool to issue |
|                    | Dispatch Unit                 | 2 per SubPart | Instruction dispatch | Reads operands, selects execution units, issues instructions |
|                    | Scoreboard                    | 1 per SubPart | Dependency tracking | Tracks register state and detects data hazards |
| **Execution-unit level** | Tensor Core | 4 per SM | Matrix multiplication | GEMM, ~1024 FLOPs/cycle, accounting for 93%+ of compute |
|                    | FP32 CUDA Cores               | 128 per SM    | Single-precision floating point | ReLU, pointwise ops, reduction |
|                    | FP64 CUDA Cores               | 64 per SM     | Double-precision floating point | Scientific computing (rarely used in ML) |
|                    | INT32 Cores                   | 64 per SM     | Integer arithmetic | Address computation, indexing, bit operations |
|                    | Load/Store Units              | 32 per SM     | Memory access | Initiates load/store requests and performs address calculation |
|                    | SFU (Special Function Unit)   | 16 per SM     | Special functions | Transcendental functions such as sin, cos, exp, rsqrt |
|                    | Texture Units                 | 4 per SM      | Texture sampling | Used for graphics rendering, occasionally for interpolation in ML |

#### Hierarchy of compute components

```
GPU
 └── GigaThread Engine (global scheduling)
      └── SM ×132
           ├── Warp Pool (up to 64 resident warps)
           └── SubPartition ×4
                ├── Warp Scheduler ──► selects warps
                ├── Dispatch Unit ×2 ──► issues instructions
                └── Execution Units
                     ├── Tensor Core (matrix multiply)
                     ├── FP32 Cores ×32 (vector arithmetic)
                     ├── INT32 Cores ×16
                     ├── FP64 Cores ×16
                     ├── LD/ST Units ×8
                     └── SFU ×4
```

#### Summary of GPU memory components

| Level | Component | Capacity (H100) | Bandwidth | Latency | Scope | Use |
|------|------|-------------|------|------|--------|------|
| **Off-chip** | HBM (device memory) | 80 GB | 3.35 TB/s | ~400 cycles | Global | Model weights, activations, large tensors |
| | L2 Cache | 50 MB | ~12 TB/s | ~100 cycles | Global | Automatically caches HBM data |
| **SM level** | SMEM (Shared Memory) | 256 KB per SM | ~33 TB/s | ~20 cycles | Shared within a block | Tile data, inter-thread communication |
| | L1 Cache | Shared with SMEM | ~33 TB/s | ~20 cycles | Private to an SM | Automatic caching (configurable partitioning) |
| | TMEM (Tensor Memory) | New in B200 | Extremely high | Extremely low | Private to a SubPart | Dedicated cache for feeding Tensor Cores |
| **Thread level** | Register File | 64K ×32bit per SM | ~80 TB/s | 1 cycle | Private to a thread | Local variables, intermediate results |
| | Local Memory | Spills to HBM | Same as HBM | High | Private to a thread | Register spill |
| **Special** | Constant Memory | 64 KB | Broadcast-optimized | ~4 cycles (cached) | Read-only global | Constant parameters, hyperparameters |
| | Texture Memory | Shared with L1 | Optimized for spatial locality | Medium | Read-only global | 2D spatial data access |

#### Memory hierarchy pyramid

```
                    ┌─────────┐
                    │ Register│  64K×32bit/SM, 1 cycle, ~80 TB/s
                    │  File   │  Thread-private
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  SMEM   │  256 KB/SM, ~20 cycles, ~33 TB/s
                    │L1 Cache │  Block-shared / automatic cache
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │L2 Cache │  50 MB, ~100 cycles, ~12 TB/s
                    │         │  Globally shared, automatically managed
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │   HBM   │  80 GB, ~400 cycles, 3.35 TB/s
                    │ (DRAM)  │  Global, persistent storage
                    └─────────┘

Capacity: Small ◄─────────────────────────────► Large
Speed:    Fast ◄─────────────────────────────► Slow
```

#### Typical usage scenarios for each memory type

| Memory | Typical use in ML | Programming model |
|------|----------------|---------|
| **Register** | Accumulators, loop variables, Tensor Core inputs/outputs | Automatically allocated, local variables |
| **SMEM** | GEMM tiling, attention K/V cache, intermediate reduction results | Explicitly declared with `__shared__` |
| **L2** | Data reused across SMs (e.g., different heads from the same batch) | Automatic, with optional hints via `cudaAccessPolicyWindow` |
| **HBM** | Weight matrices, input/output tensors, optimizer state | `cudaMalloc`, global arrays |
| **Constant** | Layer hyperparameters, lookup tables | Declared with `__constant__` |


### Understanding the working mechanism of warp and dispatch through pseudocode


```python
# ============ SM internal structure ============
class SM:
    def __init__(self):
        # Execution units (using the Ampere architecture as an example, each SM has 4 sub-partitions)
        self.sub_partitions = [SubPartition() for _ in range(4)]
        
        # Each sub-partition has its own warp scheduler + dispatch units
        
class SubPartition:
    def __init__(self):
        self.warp_scheduler = WarpScheduler()
        self.dispatch_units = [DispatchUnit(), DispatchUnit()]  # typically 2
        
        # Execution units
        self.int32_units = [INT32_ALU() for _ in range(16)]
        self.fp32_units = [FP32_ALU() for _ in range(16)]
        self.fp64_units = [FP64_ALU() for _ in range(8)]
        self.ld_st_units = [LoadStoreUnit() for _ in range(8)]
        self.sfu_units = [SpecialFuncUnit() for _ in range(4)]  # sin, cos, exp...
        self.tensor_cores = [TensorCore() for _ in range(1)]


# ============ Warp Scheduler ============
class WarpScheduler:
    """Decide which warp executes in the next cycle"""
    
    def __init__(self):
        self.warp_pool = []  # all warps managed by this scheduler (typically around 8)
        
    def select_warps_to_issue(self):
        """Select warps that can issue each cycle"""
        
        ready_warps = []
        for warp in self.warp_pool:
            if self.is_warp_eligible(warp):
                ready_warps.append(warp)
        
        # Scheduling policy: GTO (Greedy Then Oldest), LRR (Loose Round Robin), etc.
        selected = self.scheduling_policy(ready_warps)
        return selected  # may return 1-2 warps (depending on the number of dispatch units)
    
    def is_warp_eligible(self, warp):
        """Check whether a warp can be scheduled"""
        
        if warp.is_finished():
            return False
            
        # Check the scoreboard: are the instruction operands ready?
        next_inst = warp.get_next_instruction()
        if not self.scoreboard.operands_ready(warp.id, next_inst):
            return False  # data dependency, stall
            
        # Check structural hazards: is the target execution unit available?
        if not self.check_structural_hazard(next_inst):
            return False
            
        # Check whether it is waiting at a barrier
        if warp.waiting_at_barrier:
            return False
            
        return True
    
    def scheduling_policy(self, ready_warps):
        """Example scheduling policy: GTO - prefer issuing the same warp consecutively"""
        if not ready_warps:
            return []
        
        # Prefer the previously issued warp (locality)
        if self.last_issued_warp in ready_warps:
            return [self.last_issued_warp]
        
        # Otherwise select the oldest ready warp
        return [min(ready_warps, key=lambda w: w.age)]


# ============ Dispatch Unit ============
class DispatchUnit:
    """Dispatch instructions selected by the warp scheduler to execution units"""
    
    def dispatch(self, warp, instruction):
        """Dispatch an instruction to a specific execution unit"""
        
        # 1. Read operands from the Register File (data for 32 threads)
        operands = self.read_operands(warp, instruction)
        # operands contains 32 values, one per lane
        
        # 2. Select the execution unit based on the instruction type
        exec_unit = self.select_execution_unit(instruction)
        
        # 3. Issue to the execution unit
        exec_unit.issue(warp.id, warp.active_mask, instruction, operands)
        
        # 4. Update the scoreboard: mark the destination register as pending
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


# ============ Execution units ============
class FP32_ALU:
    """FP32 execution unit - SIMT execution"""
    
    def issue(self, warp_id, active_mask, instruction, operands):
        """Execute computation for 32 threads"""
        
        results = [None] * 32
        for lane in range(32):
            if active_mask & (1 << lane):  # execute only active threads
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
        
        # Write back to the register file (pipelined, may take several cycles)
        self.writeback_queue.enqueue(warp_id, instruction.dest_reg, results)


# ============ Complete per-cycle flow ============
def sm_cycle(sub_partition):
    """Pipeline operations for each clock cycle"""
    
    # Stage 1: Warp Scheduler selects warps
    selected_warps = sub_partition.warp_scheduler.select_warps_to_issue()
    
    # Stage 2: Dispatch Unit dispatches instructions
    for i, warp in enumerate(selected_warps):
        if i < len(sub_partition.dispatch_units):
            instruction = warp.fetch_next_instruction()
            sub_partition.dispatch_units[i].dispatch(warp, instruction)
            warp.pc += 1
    
    # Stage 3-N: Execution-unit pipeline execution (in parallel)
    for unit in all_execution_units(sub_partition):
        unit.pipeline_tick()
    
    # Writeback: completed results write back to the register file and update the scoreboard
    sub_partition.process_writebacks()
```

## Flowchart

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
│  │          │ Which warp is ready?                         │ │
│  │          ▼                                              │ │
│  │   ┌──────────────┐                                      │ │
│  │   │Warp Scheduler│ ──selects 1-2 eligible warps         │ │
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

| Component | Responsibility | Analogy |
| ------------------ | ---------------------- | -------------- |
| **Warp Scheduler** | Decides "who executes," checks dependencies and hazards, and chooses a policy | Dispatcher: chooses the next player to send in |
| **Dispatch Unit**  | Decides "how to execute," reads operands, selects execution units, and issues instructions | Coordinator: sends the player to the correct lane |

Cycle 1:  Warp_A: LD r1, [addr]     # initiates memory read, must wait ~400 cycles
Cycle 2:  Warp_B: ADD r2, r3, r4    # switch to B
Cycle 3:  Warp_C: MUL r5, r6, r7    # switch to C
...
Cycle 400: Warp_A: (memory returns)       # A's data has arrived
Cycle 401: Warp_A: ADD r8, r1, r9   # A resumes execution

### Importance of Tensor Cores

The numbers below use the H100 SXM scale for intuition: dense BF16 Tensor Core throughput is about 990 TFLOPs, while FP32 CUDA Core throughput is roughly 60-66 TFLOPs. The point is not a precise benchmark; it is to show where the main compute path of modern ML workloads lives.

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

The precise conclusion is: Transformer projection, MLP, and attention GEMMs should land on Tensor Cores whenever possible. CUDA Cores still handle address arithmetic, branches, elementwise ops, reductions, softmax, normalization, and glue code. For a compute-bound ML kernel, if the main path is not using Tensor Cores, lower-level scheduling tricks are usually the wrong first priority.


Ref: 
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
