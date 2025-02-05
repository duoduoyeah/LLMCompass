# LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference

**Hengrui Zhang, August Ning, Rohan Baskar Prabhakar, David Wentzlaff**  
Princeton University  
Princeton, New Jersey, USA  
{hengrui.zhang, aning, rohanbp, wentzlaf}@princeton.edu  

---

**Abstract**—The past year has witnessed the increasing popularity of Large Language Models (LLMs). Their unprecedented scale and associated high hardware cost have impeded their broader adoption, calling for efficient hardware designs. With the large hardware needed to simply run LLM inference, evaluating different hardware designs becomes a new bottleneck.

This work introduces LLMCompass1, a hardware evaluation framework for LLM inference workloads. LLMCompass is fast, accurate, versatile, and able to describe and evaluate different hardware designs. LLMCompass includes a mapper to automatically find performance-optimal mapping and scheduling. It also incorporates an area-based cost model to help architects reason about their design choices. Compared to real-world hardware, LLMCompass’s estimated latency achieves an average 10.9% error rate across various operators with various input sizes and an average 4.1% error rate for LLM inference. With LLMCompass, simulating a 4-NVIDIA A100 GPU node running GPT-3 175B inference can be done within 16 minutes on commodity hardware, including 26,400 points of the mapper’s parameter search.

With the aid of LLMCompass, this work draws architectural implications and explores new cost-effective hardware designs. By reducing the compute capability or replacing High Bandwidth Memory (HBM) with traditional DRAM, these new designs can achieve as much as 3.4x improvement in performances/cost compared to an NVIDIA A100, making them promising choices for democratizing LLMs.

---

**Keywords**—Large language model, performance model, area model, cost model, accelerator  

---

## 1. Introduction

Large Language Models (LLMs), the technology behind OpenAI ChatGPT [49], GitHub Copilot [22], and Google Bard [24], are gaining widespread attention from the whole society. The capability of LLMs is related to their model size [29], [31], and larger models [8], [11] show impressive abilities [77] compared to smaller counterparts [16], [57], with future models expected to exceed trillions of parameters [17].

This unprecedented scale of LLMs poses challenges to deployment. Serving a GPT-3 (175B parameters) inference requires a minimum of five NVIDIA A100s solely to accommodate the model parameters (in half precision). This substantial hardware cost impedes the broader adoption of LLMs and motivates computer architects to design more cost-effective hardware. We identify three challenges that exist in designing hardware for LLM inference:

1. **Lack of tools to evaluate hardware designs.** Before diving into writing the RTL code, hardware designers may want to first sketch and compare different design choices. There are many properties we want for such a hardware evaluation tool: **Fast and accurate.** Due to the intense compute and memory hardware demand required for LLM inference, the tool needs to be as fast as possible without sacrificing accuracy. **Architecturally descriptive.** This tool should be general enough to describe as fast design choices: If it only applies to a specific architecture, the design space for computer architects will be limited. **Performance-optimal.** The hardware performance is also affected by how the software is programmed (e.g., how to map the workload to the hardware). The evaluation tool should optimize this software domain to fully demonstrate the hardware capability of each design. **Cost-aware.** We also want to know how different hardware design choices affect the hardware cost to reason about cost-performance trade-offs.

Existing tools fail to meet these requirements. Roofline model analysis is fast but not accurate, and cycle-level simulators are accurate but slow. FPGA emulation is accurate and provides area statistics but requires significant engineering effort. To evaluate large-scale hardware designs in the era of LLMs, a new hardware evaluation tool is needed.

2. **Lack of knowledge on how different hardware design choices affect LLM inference performance cost.** As an emerging application, the hardware characteristics of LLMs remain to be understood. Besides the large volume of compute and memory requirements, LLMs are also unique in their autoregressive way of generating tokens. We are interested in exploring whether these properties of LLMs will change common architecture wisdom.


Lack of cost-effective hardware designs to democratize LLMs. LLMs are powerful and capable, but are cost-prohibitive to deploy. To serve GPT-3, a DGX A100 compute node can cost over $100,000 USD [46], with each NVIDIA A100 featuring 54B transistors and 80 GB of High Bandwidth Memory (HBM). This high hardware cost hinders democratizing LLMs.

In this paper, we tackle these challenges and make three main contributions.

(1) We introduce LLMCompass, a hardware evaluation framework for LLM inference workloads (Sec. III). LLM- Compass leverages the fact that mainstream ML hardware platforms share many architectural commonalities, allowing us to develop a general hardware description template for them. We also observe LLMs’ computation graphs are composed of dense operators: matrix multiplication, softmax, layer normalization, etc., all of which have a structural and hence predictable compute and memory access pattern. This allows LLMCompass to perform faster, higher-level tile-by- tile (block-by-block) simulations without losing accuracy compared to cycle-accurate simulators. The framework implements a mapper to manually manage the memory hierarchy and find the performance-optimal mapping and schedule scheme for dense workloads. LLMCompass also features a cost and area model based on public parameters to help designers reason about different design choices.

LLMCompass is validated on three commercial hardware designs: NVIDIA V100 [5], Intel A10 [M12], and Google TPUv3 [30], [45]. Compared to real-world hardware, LLM- Compass’s overall latency achieves 10.9% error rate across various operators with various input sizes, and 4.1% error rate for LLM inference. Implemented in Python, LLMCompass is still fast. It takes only 15-16 minutes to simulate a 4-A100 GPU node running GPT3-175B inference, including 26,400 rounds of the mapper’s parameter search (Figure 5, section IV, on a set of Intel Xeon Gold 6242R CPU @ 3.10GHz).

(2) We leverage LLMCompass to draw architectural implications and explore how hardware design choices affect LLM inference (Sec. IV). We find that prefill and decoding pose different hardware requirements. Prefill can significantly benefit from more compute capability and buffers, while decoding barely gains from these and is more sensitive to memory bandwidth. These insights inspire us to think about new hardware design paradigms.

(3) We propose two cost-effective hardware designs different from conventional wisdom (Sec. V). We find that today’s hardware design paradigms tend to fit massive compute capability and SRAMs in a huge die connected to high-end HBMs. We analyze the LLM inference characteristics and suggest how current hardware designs are inefficient. ① As LLM inference is mostly IO-bound, HBMs can be used to achieve low latency. However, HBM memory capacity limits the batch size, making it hard to fully utilize the massive compute capability. Based on this observation, we find that 95.3% of the original performance can still be achieved even if we prune the compute capability and buffer size by half. ② Larger batch size can significantly improve throughput as the model parameters are only read once for the whole batch. As memory capacity limits the batch size therefore limiting throughput, we propose to replace HBMs with traditional DRAM. We find that a larger batch size can compensate for the loss in memory bandwidth and can bring a 1.42× improvement in throughput and a 3.41× improvement in performance/cost.

![Fig. 2](image-url) Fig. 2: A Decoder-Only Transformer Layer with Tensor Parallelism. GPT-3 175B [8] consists of a stack of 96 such layers.

## II. BACKGROUND

### A. Large Language Models and Transformers

Large Language Models are variations of Transformer models [73] with a considerable amount of parameters that have been pre-trained on large corpora of data [40]. Today’s LLMs can have as much as one trillion parameters [17]. Compared to smaller models, larger models (e.g. GPT-3 175B [8]) showcase a remarkable set of capabilities such as emergent abilities [77] and few-shot learning [8]. This increase in model size and the consequent memory and compute requirements have posed unique challenges for hardware.

We focus on Decoder-only Transformer models [55], which is the architecture adopted by most of the LLMs today: LLaMA [70], GPTs [8], [57], Bloom [80], PaLM [11], etc. The basic building blocks of these models are Transformer layers. As illustrated in Figure 2, each layer comprises a Multi-Head Attention block followed by an MLP block. These layers are then stacked together, forming the bulk of an LLM’s memory and compute requirement. Transformers also use learned Vocabulary and Position embeddings, but for large models like GPT-3, these do not contribute significantly to either the memory or compute requirement (≤ 2%). Without losing generality, we focus on Multi-Head Attention Transformers (GPT-style). There are other variations such as Multi-Query Attention [11], Mixture-of-Experts [17], and parallel Attention and MLP [11]. LLMCompass seamlessly supports all these possible variations as they share a common set of operators.

### B. LLM Inference

Given an input prompt and the required number of output tokens, LLM inference can be divided into two stages [56]. ① Prefill: Processing the input prompt and computing the KV cache. The Key Value (KV) cache refers to the stored Key and Value tensors of the Attention block in each layer [56].

---

1081


![Figure 3: LLMCompass' Hardware Description Template. In this example, each device has 2 cores and each core has 2 lanes.](Figure)

② Decoding: Generating output tokens one after another in an auto-regressive manner: The Key and Value of the newly generated token will be concatenated to the KV cache and used for generating the next token. The latency of prefill and decoding is mostly determined by the input and output sequence lengths, respectively. In prefill, as the entire input sequence needs to be multiplied by all the parameters, it is usually bounded by compute. In decoding, each new token needs to be multiplied by all the parameters and concatenated to the KV cache, so decoding is usually bounded by reading parameters and KV cache.

Latency and throughput are the key metrics to evaluate LLM inference systems. For interactive use cases such as chatbots [49], it is imperative to optimize latency. For background data processing use cases such as data warping [42] or form processing [9], throughput is more important. The tradeoff between latency and throughput is determined by batch size: larger batch increases throughput at the cost of higher latency.

C. Parallelizing LLM Inference

Due to the large volume of compute and memory operations, it is beneficial to parallelize LLM inference across multiple devices. This leads to much better performance and can be necessary if the model’s parameters along with the KV cache do not fit in a single device’s memory. For LLM inference, there are two model parallelization schemes: pipeline parallelism and tensor parallelism. In pipeline parallelism, different layers of the model are grouped into sequential partitions and assigned to different devices like a hardware pipeline. This scheme has the effect of considerably increasing throughput at the expense of increased latency. On the other hand, tensor parallelism, as proposed by Megatron-LM [64], partitions each layer of the model across the available devices, thereby decreasing latency at the cost of frequent device-device communication and synchronization. As shown in Figure 2, this scheme requires two all-reduce for each Transformer layer, one after the Attention block and another after the MLP block.

III. LLMCompass

An overview of LLMCompass (Large Language Model Computation Performance and Area Synthesis) is shown in Figure 1. To evaluate the performance (e.g., throughput and latency) of running a Transformer-based large language model on a hardware system, two inputs are needed: the computational graph of the LLM and a hardware description (Section III-A). Given the input, the performance model (Section III-B) generates a performance report. The mapper conducts a parameter search along with the architecture simulator to find the best mapping and scheduling choice. At the same time, the area model (Section III-D) generates the area and cost report.

A. Hardware Description Template

The hardware description template of LLMCompass is introduced below, as shown in Figure 3:

- A system (e.g., a DGX node) is composed of multiple devices connected through a device-device interconnect (e.g., NVLink or Infinity Link).
- Each device (e.g., a GPU) is composed of multiple cores, a shared global buffer, and an off-chip main memory. The global buffer (e.g., L2 cache in NVIDIA GPUs) is connected to the main memory, device-device interconnect, and all the cores.
- Each core (e.g., a Stream Multiprocessor in NVIDIA GPUs) can have multiple lanes sharing a local buffer (e.g., L1 cache in NVIDIA GPUs). The local buffer is connected to the global buffer through the on-chip interconnect.
- Each lane is independent from each other and has its own vector unit, systolic array, registers and control logic.

TABLE I: Examples of LLMCompass’s Hardware Description
| Key Specifications        | NVIDIA A100 [48] | AMD MI210 [2] | Google TPUv3* [45] |
|---------------------------|------------------|---------------|--------------------|
| Frequency (MHz)           | 1104            | 1700          | 940                |
| Core count                | 108             | 104           | 2                  |
| Lane count                | 4               | 4             | 4 × 128            |
| Systolic array            | 16 × 16         | 16 × 16       | 128 × 128          |
| Local buffer (KB)         | 192             | 800           | -                  |
| Global buffer (MB)        | 40              | 8             | 1834               |
| Global buffer (cycles)    | 5120            | 496           | 490                |
| Memory bandwidth (TB/s)   | 2               | 1.6           | -                  |
| Memory capacity (GB)      | 80              | 64            | -                  |
| Device-device bandwidth (GB/s) | 600      | 300           | 162.5              |

*One TPUv3 core. Each TPUv3 chip has two TPUv3 cores. TPUv3 cores within the same chip are connected by internal links.


![Fig. 4: Visualization of a Matrix Multiplication in LLMCompass as in Section III-B1.](image-url)

In existing devices, the local and global buffers are usually on-chip SRAM: cache, scratchpad, or a combination of both. 
LLMCompass doesn’t distinguish between cache and scratchpad because the memory is explicitly managed by the mapper.
We believe this assumption does not lose generality as a highly optimized library will also carefully manage the memory.
The main memory is usually off-chip DRAM: HBM, DDR Memory, CXL memory, etc., all of which can be described by
our parameterized hardware description template.

We find this hardware description is general enough to describe the mainstream machine learning platforms of today:
NVIDIA GPUs, AMD GPUs, and Google TPUs, as shown in Table I with a sample of key specifications listed. It is also
flexible enough to explore future architectures.

B. Performance Model
The computational graph of a Transformer is composed of a
stack of Transformer layers. Each layer is composed of a
series of operators, including matrix multiplication (Matmul),
Softmax, layer normalization (LayerNorm), and activation
functions (e.g., GELU [28] as in GPTs [8], [57]). In a multi-
device setup, communication primitives such as all-reduce
operators are also needed to perform tensor parallelism. The
key challenge is how to simulate the performance of different
operators and communication primitives on a given hardware
system - this requires knowledge about the hardware and
how to map and schedule operators on a multi-level compute
system with a multi-level memory hierarchy.

To solve this, LLMCompass introduces a mapper and an
architecture simulator to build a performance model. Concep-
tually, we simulate running an operator on the chosen hardware
in a recursive manner: we first partition the problem into
smaller sub-problems that can fit in the global buffer. The sub-
problem is then divided into smaller sub-sub-problems that can
fit in each core’s local buffer. The partitioning, mapping, and
scheduling are generated by the mapper and a parameter search
is conducted to find the optimal mapping and scheduling.
LLMCompass always tries to find the performance-optimal
mapping to fully demonstrate the hardware capability.

1) Matrix Multiplication: The process of simulating a
matrix multiplication is visualized in Figure 4. A is a M x K
matrix with M rows and K columns. Similarly, B and C are
K x N and N x M matrices respectively, a generalized matrix
multiplication is defined as C = A B + C.

From main memory to global buffer: To maximize data
reuse, matrix multiplication is usually calculated in a tile-by-
tile manner [34]. As shown on the left of Figure 4, matrix
A, B, and C are divided into tiles small enough to fit into
the global buffer. In each step, one A_tile_{m,k}, B_tile_{k,n}
and C_tile_{n,m} are read into the global buffer, the cores then
perform the computation, and the results are written back.

From global buffer to local buffer: With tiles inside the
global buffer, we now need to parallelize the computation of
C_tile_{n,m} = A_tile_{n,m}B_tile_{k,n} + C_tile_{n,m} on multiple
cores. As shown in the middle of Figure 4, these tiles are
further divided into smaller sub-tiles to fit in each core’s local
buffer. It then becomes a scheduling problem to map sub-tiles
onto cores.

The right of Figure 4 shows two possible schedule schemes:


- **Schedule Scheme 1:** Different cores working on different `C_subtitle`s in the same column. At wave 0, as core 0 and core 1 both need to read the same `B_subtitle`, their memory access to the global buffer should be merged. In our simulator, this memory access merging is automatically identified and taken care of. As the same core keeps updating the same `C_subtitle`, there is no need to first write the partial result and then read it from the global buffer. This Read-After-Write dependency is also automatically taken care of by the simulator.  
- **Schedule Scheme 2:** Different cores working on the same `C_subtitle`. Core 0 and core 1 first read the data and calculate the partial results, then perform a reduction and write back the final results.

In reality, with more cores and more tiles, the schedule space can be more complicated than the example shown in Figure 4.

From local buffer to tiles: Similarly, within each core, the sub-tiles are further partitioned as sub-sub-tiles to be mapped to lanes sharing a local buffer. After that, the sub-sub-tiles are finally passed to the systolic arrays. LLMCompass leverages SCALE-Sim [61], [62], a cycle-level systolic array simulator, to mimic the behavior of a systolic array and get the cycle count. LLMCompass caches the results of SCALE-Sim into a look-up table to avoid duplicated simulation. A reduction will be performed by the vector unit if needed.

- **Mapper:** A parameter search is performed by the mapper to determine the best tiling scheme and schedule scheme, based on the cache (with possible copy), compute, and software pipelines (double buffering) at each level of the memory hierarchy as scheduling options. The downside of enabling software pipeline is that it requires extra buffer space so the maximal tile size will be reduced, causing potentially lower utilization of systolic arrays. However, we find software pipeline to be beneficial in most cases.

2. **Communication Primitives:** We use the link model as in AHEAD [1] and LogGP [4]. Suppose `L` is the link latency, `O` is the additional overhead associated with the data transfer, and `B` is the link bandwidth. The latency `T` to transfer `n` bytes of data through a link is expressed in Equation 1 and 2:

   ```
   T = L + O + ⌈ n / MaxPayload ⌉ * Fit_size + n / B
   ```

   On top of this, we implement ring all-reduce [52], which is a bandwidth-optimal all-reduce algorithm. We use a 16-byte `Fit_size` and a 256-byte `MaxPayload` based on NVLINKs [18]. We don’t model more communication primitives as LLM inference only requires all-reduce for tensor parallelism and peer-to-peer for pipeline parallelism.

3. **Other Operators:** We also model Softmax, LayerNorm, and GELU following a similar methodology as in Serciontinelli-LB. The differences are as follows: 0) These operators have fewer dimensions and are therefore simpler. Softmax and LayerNorm operate on two-dimensional data, and GELU operates on one-dimensional data, while Matmul operates on three-dimensional data. As each dimension requires tiling and scheduling, the mapper search space is much smaller. 2) They do not use systolic arrays. 3) Softmax and LayerNorm involves reductions to calculate the sum, mean, or max. Therefore, the schedule scheme needs to consider that the reduction can be either performed within one core or might be splitted across different cores. For the reduction within each core, a reduction tree is implemented. Inter-core reduction is implemented with atomic operations. Softmax is implemented with the online algorithm [39]. GELU is approximated with `tanh` [28].

6. **Performance Model Validation**

In this section, we validate our framework against three real hardware platforms: (1) a datacenter GPU node with 4 NVIDIA A100 SXM4 GPUs (80 GB fully connected by NVLinks; (2) a Google TPU node with 8 TPUv3 cores connected in a 2D torus topology; (3) an AMD MI210 GPU3. The results are shown in Figure 5. For NVIDIA GPUs, CUDA 11.7 and PyTorch 2.0 are used to benchmark operators in half precision (FP16) with `torch.compile` enabled for LayerNorm and GELU to maximize performance. Communication primitive all-reduce is benchmarked with nccl-tests [43], a communication primitive performance benchmark for NVIDIA GPUs. For Google TPUs, JAX 0.4.18 is used to benchmark operators and communication primitives. Due to the hardware feature of TPUs, Matmul is benchmarked in bfloat16 (BF16) and all the other operators are in FP32. For AMD GPU, ROCm 5.4.2 [38] is used to benchmark operators with FP16 for Matmul and FP32 for other operators. The kernel launch overhead including the framework overhead is measured by running the operator with input of size 1.

As shown in Figure 5, for Matmul, Softmax, LayerNorm, GELU, and all-reduce, LLMCompass achieves an average error rate of 9.0%, 12.0%, 13.8%, 5.0%, and 14.9% respectively. For LLM inference, LLMCompass achieves an average error rate of 0.69% and 7.5% for prefill and decoding respectively. On average, LLMCompass achieves a 10.9% error rate for different operators at various input sizes and a 4.1% error rate across the prefill and decoding stages.

GELU is more accurate than other operators because it is element-wise and easy to simulate. Layernorm and Softmax are less accurate because of the reduction involved. All-reduce is less accurate probably because of unidealized hardware. Matrix multiplication is accurate (except for small ones on AMD MI210 as in Figure 5b) because it is highly optimized on those hardware platforms. As matrix multiplication is the dominant part of most of the models today, a validity of performance across different types of models can be achieved.

Although not a perfect match to real-world hardware, LLMCompass is able to show a similar trend that a naive reduction model fails to show. For example, in Figure 5e, as the reduction dimension of LayerNorm increases to an extreme, the throughput should drop due to the increasing reduction cost. LLMCompass is able to catch this trend.

3 We set the frequency to 1400 MHz to avoid frequency fluctuation

---

Authorized licensed use limited to: OREGON STATE UNIV. Downloaded on January 27,2025 at 12:15:57 UTC from IEEE Xplore. Restrictions apply.

![Figure displaying performance metrics for various hardware and software setups.](#)

#### Figure 5: Performance Model Validations.

1. Matmul takes a M × K (M rows and K columns) and a K × N matrix as input. 12288 is the model dimension of GPT-3 [8]. 

2. Softmax and LayerNorm take a M × N matrix and perform normalization on the N dimension.

3. Prefill latency is measured by running one layer of GPT-3 with batch size 8 and sequence length 2048. Decoding latency is per GPT-3 layer per output token and is measured by the latency of generating the 1024th output token with batch size 8 and input sequence length 2048. For (a)-(g), a single GPU/TPU device is used. For (h)-(l), the A100 GPU node and 8-TPUv3-Core TPU node are used with tensor parallelism.

---

### Observations:
- LLMCompass's results are totally interpretable without incorporating any fudge factor and we believe this interpretability is more important than perfectly matched results. Here are some possible causes of the mismatch between LLMCompass and real hardware:
  - **Lack of hardware knowledge.** We have little knowledge about the micro-architecture of GPUs and TPUs (e.g., hardware pipeline design or scheduler design). With a large input size, the hardware is well utilized and some overhead can be hidden. However, with a small input size, it’s hard to hide the overhead and micro-architecture details affect performance significantly. Also, the Tensor Cores in NVIDIA GPUs and Matrix Cores in AMD GPUs are simulated as systolic arrays in LLMCompass, which may not be true in reality.
  - **Lack of software knowledge.** We don’t know how operators and communication primitives are implemented on these platforms as they are closed-source libraries. We conduct a thorough parameter search for each input size to maximize performance, but in reality those libraries probably use heuristics to determine mapping and scheduling, which may not be optimal at all input sizes (e.g., we find that for a Matmul with M = 64 and N = K = 12288, AMD MI210 is less than 25% as fast as A100 due to suboptimal matrix layout selection, but with M = 12288, N = K = 64, the MI210 launches significantly faster kernels).

---

### Figures:

- (a)-(d): Performance metrics for matrix multiplication across different configurations.
- (e)-(g): Comparing GELU, LayerNorm, and Softmax across hardware and software configurations.
- (h)-(l): Latency metrics for GPT prefill and decoding processes on GPU and TPU setups.

---

#### Detailed Analysis:

From the figures, you can observe:
- NVIDIA A100 typically outperforms AMD MI210 for most configurations.
- Prefill latency and decoding latency are quantitatively illustrated, validating performance modeling versus real hardware behavior.
- Differences highlight the limitations and approximations of emulation models like LLMCompass and emphasize factors like hardware/software inhomogeneities.

---


of its roofline performance while a NVIDIA A100 can achieve 50% of its roofline performance.). Also, some key information is not available. For example, we cannot find the packet format for TPU-TPU communication and have to use the NVLink packet format instead. 
- Non-ideal hardware. LLMCompass assumes a fixed fre­quency, but when testing real-world hardware, we have no control over the frequency of the datacenter GPU or TPU nodes. LLMCompass also assumes band width can be utilized at full rate, but in reality there may be some other overhead (e.g., error correction code).

D. Area and Cost Model

As chip designers increase die area to improve single chip performance, fewer chips fit per wafer and may also risk decreased yield, leading to increased costs. LLMCompass incorporates area and cost models to allow designers to reason about these performance-area trade-offs. These models use the provided hardware description with estimated transistor counts and/or die areas from known components to find the total device die area - our methodology is explained as follows.

Within each core’s lanes, we estimate the vector units’ and systolic arrays’ transistor counts from open-source designs, tape-outs, and generators [20], [38], [83]. We estimate each lane’s register file’s area overhead based on an empirical area model [85]. For the local buffer shared amongst lanes in each core as well as the global buffer shared amongst cores, we estimate the buffer’s area overhead using known reports, when CACTI [41] and scale results down to a 7nm process. For memory and device-device interconnect, we estimate PHY and controller area based on annotated A100 and MI210 die photos [53], [65]. In our calculations, the controller area scales based on the process node, but the PHY area remains fixed as they do not scale well due to internal analog devices.

We account for extra per lane overheads (e.g., control signals) by calculating the core area using our model and taking the difference from the expected die areas taken from annotated photos. We then divide the overhead per lane, per scheduler width (32 in AI100s, 16 in MI210). Similarly, we account for extra per core overheads (e.g., core-to-core crossbars) by calculating the expected die area with our model and splitting the area between the cores. These per-lane and per-core overhead estimates are averaged between AMD and NVIDIA chips.

To estimate cost, LLMCompass uses supply chain model­ing [44] for wafer costs to calculate per-die costs. These per-die costs do not incorporate any IP, masks, or packaging costs. For memory costs, we use average DRAM spot prices for DDR [71] and consumer estimates for HBM2e [35].

Table II shows a sample of the transistor counts and corresponding 7nm die areas of the parameters used in the area model. Using their respective architecture white papers, we model GA100 [48] (the die used in NVIDIA A100) and Aldebaran [2] (the die used in AMD MI210) dies to estimate their total die areas, shown in Fig. 6a. For the area model experiments, LLMCompass’ area model estimates for

```
TABLE II: A Sample of Area Model Parameters (7nm)

| Parameter               | Transistor Count | 7nm Area (um^2) |
|-------------------------|------------------|-----------------|
| 64 Bit Floating Point Unit | 683300           | 718           |
| 32 Bit Int ALU           | 172700           | 135           |
| Per Core Overhead       | 9902000          | 60814         |
| Pore Core Overhead      | 43200000         | 464000        |
| 1024 Bit HBM2e Control  | 552743000        | 5707000       |
| 1024 Bit HBM2e PHY      | -                | 10450000      |

(a) Die Area Breakdown of NVIDIA GA100 and AMD Aldebaran.

(b) Core Area Breakdown (Stream Multiprocessor for NVIDIA GPUS and Compute Unit for AMD GPUs).

    ![Bar charts showing die area breakdown: Real and Simulated GA100 and Aldebaran, categorized as Local buffer, Register file, Systolic array, and Control logic.](bar-charts.png)

Fig. 6: Area Model Validations.

With LLMCompass, we are able to conduct a design space exploration and shed light on how to design efficient hardware systems for LLM inference. In this section, we use LLMCom- pass to study how different compute system configurations, memory bandwidth, and buffer sizes affect LLM inference per­formance and draw architectural implications. These insights inspire us to propose new designs as in Section V.

IV. ARCHITECTURAL IMPLICATIONS

With LLMCompass, we are able to conduct a design space exploration and shed light on how to design efficient hardware systems for LLM inference. In this section, we use LLMCom- pass to study how different compute system configurations, memory bandwidth, and buffer sizes affect LLM inference per­formance and draw architectural implications. These insights inspire us to propose new designs as in Section V.

A. Experimental Setup

For all the unmentioned specifications, we use the specifi­cations of an NVIDIA A100 (as in Table I) and 4-way tensor parallelism. Preﬁll latency (also know as TPTF, time to first token) is measured by running one GPT-3 layer with batch size 8 (a balancing point between latency and throughput) and input sequence length 2048 (a medium-long sequence for GPT-3). Decoding latency (also know as TBT, time between tokens) is measured as the latency of generating the 1024th output token when running one GPT-3 layer with batch size 8 and input sequence length 2048. We use FP16 for all the operations.

![Die Area Breakdown and Core Area Breakdown](Fig6.png)

1086



| TABLE III: Five Compute System Designs. |          |      |      |      |      |
|------------------------------------------|----------|------|------|------|------|
| Specifications                           | A        | B    | C    | D    | E    |
| Core count                               | 128      | 128  | 128  | 32   | 1    |
| Latent count                             | 4        | 4    | 1    | 1    | 1    |
| Vector width                             | 8        | 32   | 128  | 512  | 2048 |
| Systolic array                           | 8 x 8    | 16 x 16 | 32 x 32 | 64 x 64 | 128 x 128 |
| Local buffer (KiB)                       | 192      | 192  | 192  | 768  | 3072  |

![Graph (a)](/assets/graph_prefill_latency.png)
![Graph (b)](/assets/graph_decoding_latency.png)

Fig. 7: Impact of Compute System Design on Performance.

B. Compute System

We test five different compute system designs as shown in Table III. From A to E, we increase each core's systolic array, vector unit, and local buffer capacities. B represents a full GA100. We keep B, C, D, and E to have the same total compute capability and total buffer size to compare the design choice of fewer big cores or more tiny cores. Configuration A only has a quarter of the compute capability compared to others. All the designs have the same amount of total buffer size and register file size scales with vector width.

Figure 7 shows prefill and decoding latencies for these designs. Compared to the GA100, design A has 3.25x higher prefill latency but is only 0.1% slower at decoding and uses only 57.8% of the area. Design E with the largest cores see prefill and decoding latency increase by 12.4% and 1.9% respectively, but can reduce die area up to 7.7%.

Analysis: For the prefill stage, B is much faster than A because prefill is compute-bound. As per core systolic arrays and vector units scale, the tile size needs to increase to fully utilize larger computing units. Bigger tiles can cause more padding as the prefetch size needs to be quantized to the tile size and hardware size. Although large systolic arrays and vector units can be more area-efficient, they are harder to schedule and fully utilize.

Since decoding is IO-bound, increasing compute capability barely helps, which explains why A and B have similar performance. As the matrix multiplications during decoding are narrow (e.g. 16 x 12288), it is even harder to fully utilize larger systolic arrays/vector units and performance degrades.

![Graph (a)](/assets/graph_memory_bandwidth_prefill.png)
![Graph (b)](/assets/graph_memory_bandwidth_decoding.png)

Fig. 8: Impact of Memory Bandwidth on Performance.

**Implications:**

① Increasing compute capability significantly helps prefill but barely helps decoding.

② Larger systolic arrays and vector units are more area-efficient but harder to fully utilize.

C. Main Memory

As main memory capacity is considered more of a constraint (enough capacity is required to hold the parameters and KV cache), we will focus on the impact of main memory bandwidth. Figure 8 details the performance results for sweeping memory bandwidth from 400 to 3200 GB/s. For prefill, increasing memory bandwidth from 800GB/s to 2000GB/s reduces latency by 14.3%, with further increasing to 3200GB/s has a marginal performance gain of 3.5%. For decoding, increasing from 800GB/s to 2000GB/s has a speedup of 1.88x, and further increasing to 3200GB/s brings another 26% gain.

**Analysis:** In the prefill stage, Matmuls are significantly faster when increasing memory bandwidth from 400GB/s to 800GB/s. Further increasing bandwidth does not significantly affect Matmul performance as it becomes compute-bound. For IO-bound GELU, LayerNorm, and Softmax, larger memory bandwidth realizes significant speedup.

In the decoding stage, Matmuls are significantly faster with increased memory bandwidth, mainly because they are narrow (turn into vector-matrix multiplication at batch size 1) and IO-bound. In this stage, GELU, LayerNorm, and Softmax have a small input size. They are dominated by kernel launch overhead and barely affected by memory bandwidth.

③ Decoding is much more sensitive to memory bandwidth than prefill.


![Latency](path/to/image)

(a) Prefill Latency (TTFT) per GPT-3 Layer.

(b) Decoding Latency (TBT) per GPT-3 Layer per Output Token.

Fig. 9: Impact of Local Buffer Size on Performance.

D. Local and Global Buffer

**Local Buffer.** We fix the hardware specifications to an NVIDIA A100 (as in Table I) and sweep local buffer size. The results are shown in Figure 9. For prefill, increasing the local buffer size from 64KB to 192KB improves the performance by 18.0% while increasing the area by 5.8%. Further increasing to 1024KB has a negligible performance gain of only 0.2% at the cost of 28.8% bigger area. For the decoding stage, increasing the local buffer size from 64KB to 1024KB only increases the performance by 0.5%.

**Analysis:** The reduced prefill latency with larger local buffers is mainly because of reduced matrix multiplication latencies. A larger local buffer enables larger matrix tiles and therefore higher systolic array utilization rate. A local buffer size of 192KB is just enough for matrix multiplication in 128×128×128 at FP16 with double buffering technique. It can fully utilize the 16 × 16 systolic arrays, shedding some insight on the NVIDIA A100's design choices. Increasing local buffer size when the systolic array is already fully utilized leads to marginal performance gains. For decoding stage, increasing local buffer size does not help because it’s IO-bound.

**Global Buffer.** The performance trends for global buffer size are similar to Figure 9. Increasing the global buffer size from 10MB to 40MB speeds up prefill by 11.8% while increasing area by 9.6%. Further increasing to 80MB only brings a performance gain of 0.1% at the cost of 11.7% bigger area. For decoding, increasing global buffer size from 10MB to 80MB has a performance gain of only 0.7%.

**Analysis:** Larger global buffers enable larger matrix tiles, increasing systolic array utilization and data reuse at the global buffer level. Similarly, increasing global buffer size has diminishing returns once the systolic arrays are saturated. The decoding stage is not bounded by computation so it barely benefits from the larger global buffer.

TABLE IV: Comparison with NVIDIA GA100

| Specifications     | Latency Design | GA100 Design | Throughput Design |
|--------------------|----------------|--------------|-------------------|
| Core count         | 64             | 128          | 64                |
| Lane count         | 4              | 4            | 4                 |
| Vector width       | 32             | 32           | 32                |
| Systolic array     | 16 x 16        | 16 x 16      | 32 x 32           |
| Local buffer (KB)  | 192            | 192          | 768               |
| Global buffer (MB) | 24             | 48           | 48                |
| Global buffer (bytes/cycle) | 2560  | 5120         | 5120              |
| Memory bandwidth (TB/s) | 2         | 1            | 1                 |
| Memory capacity (GB) | 80           | 80           | 512               |
| Memory protocol    | HBM2E          | HBM2E        | PCIE 5.0/CXL      |
| Die area (TSMC 7nm, ㎟) | 478       | 826          | 787               |
| Normalized performance | 0.95      | 1            | 1.41              |
| Estimated die cost | 580           | 551          | 5142              |
| Estimated memory cost | 650         | 656          | 5456              |
| Estimated total cost | 5640         | 5711         | 57696             |
| Normalized performance/cost | 1.06 | 1            | 3.41              |

> 🎯 **Large buffers help prefill but not decoding.**
> 🎯 **Buffers should be large enough to fully utilize the systolic arrays.**

V. Efficient Hardware Design with LLMCompass

Ideally, efficient hardware design will optimize for both performance and cost. This section draws from the insights in Section IV and proposes two efficient hardware designs: a latency-oriented design and a throughput-oriented design. Both of these improve the performance-cost tradeoff while maintaining or improving performance. The key specifications are shown in Table IV. All the other specifications (e.g., frequency, register file size, device-device interconnect, kernel launch overhead, and framework overhead etc.) are the same as an NVIDIA GA100 for fair comparison.

A. Latency-Oriented Design

LLM inference latency means the total time between receiving the request and generating the last token. It is a critical metric for interactive use cases like chatbots. It is composed of **prefill latency**, the time to process the input sequence, and **decoding latency**, the time to generate the output sequence in an auto-regressive way. Inference latency is usually dominated by **decoding** unless the input sequence is much longer than the output sequence. **Decoding** is IO-intensive and is mostly bounded by reading model parameters and KV cache.

**Observation:** As latency is mostly IO-bound, memory bandwidth is the key to reducing latency, making HBM the best choice. However, due to the capacity limit of HBM, the batch size cannot be too large; the size of the KV cache and intermediate values is proportional to batch size. Therefore, the massive compute capability is not fully utilized.

**Proposal:** We propose an efficient latency-oriented design pruning half of the compute capability while using the same memory system as a GA100, as shown in the left of Table IV.

**Results:** Compared to an NVIDIA GA100, the die area is reduced by 42.1% while keeping 95.3% of the performance on average. The results are shown in Figure 10.


![Fig. 10: End-to-End Performance of Latency-Oriented Design Normalized to GA100. Performance metric: inverse of latency (higher is better). Settings: batch size*16, 4-way tensor parallelism, running 48 GPT-3 layers (half of GPT-3).](image)

![Fig. 11: TTFT and TBT of Latency-Oriented Design Normalized to GA100. Reducing compute capability by half barely hurts TBT but will bring 1.82x slowdown for TTFT.](image)

**Discussion:** Due to the IO-bound decoding stage, the over-provisioned A100 is not able to provide a meaningful improved inference performance compared to our latency-oriented design. As shown in Figure 11, our implementation achieves identical decoding performance as a GA100. The GA100 is an enormous die and is suboptimal to yield issues - A100 dies are already binned to have 108 functioning SMs out of 128. Our latency-oriented design shows that even with half the cores and SRAM disabled, the device can still achieve similar performance. This may motivate designers to salvage previously deemed faulty chips and manufacture them into separate products focused on LLM inference.

Pruning the compute capability only hurts the compute-bound prefill performance. As prefill is more dominant at long input sequence and short output sequence, the performance degradation will be more visible under these cases, which explains why we only achieve 80% of the GA100 performance at input length 2048 and output length 256. With a smaller input length and larger output length, our pruned latency-aware design can achieve 99% the performance as GA100.

---

### B. Throughput-Oriented Design

For background use cases such as form processing or data wrangling, throughput can be more important than latency. There are generally two ways to improve throughput:

1. **Decrease latency** - As latency is mostly IO-bound by reading parameters and KV cache, the best way to improve latency is to further improve memory bandwidth. As HBM is already expensive, this may not be achieved without increasing cost.
2. **Increase batch size** - Generally, larger batch sizes are more efficient for throughput because the parameters are only read once for the whole batch. Larger batch sizes can also improve the hardware utilization rate. The downside is that a larger batch size consumes more compute power and increases KV cache sizes.

**Observation:** Increasing batch size is a more efficient way to improve throughput compared to decrease latency, which requires expensive high-end HBMs or even SRAMs. With a larger batch size, more memory capacity is needed to hold the larger KV cache and intermediate values.

**Proposal:** We propose a throughput-oriented design as shown in the right of Table IV. To hold larger batches, we use 512GB of DRAM powered by 256 PCIe 5.0 channels with an aggregated memory bandwidth of 11TB/s. (According to our area model, an 800mm² die’s perimeter is able to fit around 400 PCIe 5.0 channels.) Considering the high cost and limited capacity of HBMs, this design is more cost-effective. With larger batch sizes comes a greater need for compute capability, so we quadruple the systolic arrays and the local buffer. We halve the core count and vector unit to maintain a similar die area as GA100.

**Results:** Compared to an NVIDIA GA100, the die area is slightly smaller and the throughput is improved by 1.42x on average. The results are shown in Figure 12. By replacing HBMs with traditional DRAMs, the cost is reduced by 58.3%, making a total of a 3.41x gain in performance/cost.

![Fig. 12: End-to-End Performance of Throughput-Oriented Design. Performance metric: throughput. Settings: largest batch size within memory capacity, 8-way pipeline parallelism where each device runs 12 GPT-3 layers (1/8 of GPT-3).](image)

---

**Authorized licensed use limited to: OREGON STATE UNIV. Downloaded on January 27, 2025 at 12:15:57 UTC from IEEE Xplore. Restrictions apply.**


![Design Space Exploration with LLMCompass](https://via.placeholder.com/200)

Fig. 13: Design Space Exploration with LLMCompass. The proposed latency-optimized design and throughput-optimized design are marked in red. Throughput and latency are normalized to a 4-GA100 node. Sweep parameters: computer system design, buffer size, memory type and capacity. Settings: largest batch size within the memory capacity, input length 1024, output length 1024, 4-way tensor parallelism, running 48 GPT-3 layers (half of GPT-3). It took 84 minutes to collect all the data points on one Intel Xeon Gold 6242R CPU @ 3.10GHz.

---

**Discussion:** Our design has 6.4x the memory capacity of GA100, which allows more than 12x bigger batch size after subtracting the fixed space occupied by model parameters. Ideally, with half the bandwidth of a GA100, this configuration can achieve more than 6x improvement in throughput. However, batching only reduces model parameter accesses but not KV cache reads. With a much larger batch, KV cache accesses are amortized. In practice, it is hard to achieve the benefits of batching. As input length and output length increase, throughput decreases due to more KV cache accesses.

From a latency perspective, this throughput-oriented design may not be promising: its latency is 9.21x worse than GA100 on average. While model parameters are only read once for each batch, a larger batch size means more KV cache and intermediate values to read. In LLM inference, there is no free lunch between latency and throughput.

---

### C. Design Space Exploration

With LLMCompass’ speed, we are able to conduct design space exploration with different hardware design choices. As shown in Figure 13, four different memory designs with different bandwidths and capacities as well as various core counts and core designs are explored.

Figure 13 indicates our proposed latency-oriented design and throughput-oriented design are around the sweet point. Decreasing compute capability too much can hurt performance due to the compute-intensive prefill stage. Increasing memory capacity also has diminishing returns as larger batches increase KV cache accesses.

---

## VI. DISCUSSION

### 1) Hardware designs that can/cannot be modeled by LLMCompass:
LLMCompass covers the dominant hardware platforms for LLMs today: NVIDIA GPUs, AMD GPUs, and Google TPUs, and can be extended to newer architectures with no/little change to the code, thanks to its generic hardware template and automatic mapping exploration. In LLMCompass, users only need to describe their design and do not need to recalibrate LLMCompass for each new design.

For the three real-world designs evaluated in the paper, we use the same code for performance and area modeling. LLMCompass can be seamlessly extended to newer architectures such as NVIDIA H100. As a train/test setup, we asked our collaborators to validate LLMCompass on an NVIDIA RTX A6000 without changing any code, and LLMCompass achieves within 2.5% error rate for LLM inference workloads.

LLMCompass does not incorporate network modeling, and therefore cannot accurately model Cerebras wafer-scale processors, which have 850K cores and are more like a distributed system where inter-core communication mechanisms play a key role. To model Cerebras-like designs, LLMCompass can aid in existing network models [50], [59], [79].

LLMCompass is designed for throughput-oriented fault-tolerant machines so it cannot model CPUs accurately due to its latency-sensitive nature and complicated control flow.

---

### 2) Other optimization techniques:
LLMCompass can be extended to a variety of optimization techniques. To support operator fusion like FlashAttention [13], users can implement a simulated fused operator based on the simulation code for its individual operators. We do not explore operator fusion in this paper as many of them are specific to NVIDIA GPUs and we are not sure whether they can be applied to other hardware platforms such as Google TPUs.

LLMCompass can also support batched training techniques. For example, ORCA-style continuous batching [82], SARATHI-style [3] chunked prefills, and Splitwise-style [54] phase splitting can be supported by wrapping a scheduling function on top of LLMCompass.

In this paper, we choose to use request-level batching with different input and output lengths (as in Figure 10, 11, and 12), as it is how NVIDIA benchmarks TensorRT-LLM [68].

---

## VII. RELATED WORK

### A. Evaluating Large-scale Hardware Design

Evaluating the various characteristics of a hardware design, including performance, area, and cost, is extremely useful for hardware designers. To this end, the options are as follows:

- **Roofline Model Analysis [78]:** Roofline models are analytical, fast to evaluate, and can be applied to various architectures for performance comparison. However, they can be overly optimistic relative to actual hardware capabilities.
  
- **Cycle-level Simulation [6], [7], [21], [23], [25], [32], [33], [51], [61], [62], [66], [72]:** With a typical simulation rate of less than 100K instructions per second, cycle-level simulators become infeasible for design space exploration of LLM scale workloads. As these simulators are often designed for specific architectures, it is hard to describe a hardware design very different from its design purpose (e.g., it’s almost impossible to use GPGPU-sim [6] to evaluate a TPU-like design because it relies on the GPU ISA). These simulators often require the user to provide the program for evaluation. If the software program is not optimized, it may lead to unfair comparisons.


FPGA Emulation. Another way is to implement the design in RTL code and emulate it on FPGAs. The RTL code can be either handwritten or generated by accelerator generators [20], [47], [69], [74]. Although the emulation is fast, the synthesis process may take a long time, and users are responsible for mapping their workloads to the hardware. Additionally, users need to repeat this whole process to evaluate a new design.

Comparison. LLMCompass is suitable for pre-silicon design space exploration before diving into more detailed cycle-level simulation or FPGA emulation. Compared with roofline model, LLMCompass is more accurate. As opposed to simulating in a cycle-by-cycle manner, LLMCompass is much faster by leveraging the insights that operators in LLMs follow a highly regular and predictable pattern. Cycle-level simulators are usually tightly bonded to specific architectures. For example, GPGPU-sim [6], [32] only supports a subset of NVIDIA architectures and does not offer official support for newer NVIDIA Ampere GPUs like A100. We could not find an existing simulator that models NVIDIA A100, AMD M210, and Google TPUv3. Compared with FPGA emulation, LLMCompass is significantly less engineering-intensive.

LLMCompass can complement FPGA emulation. Designers can perform initial design space exploration before incurring the heavy costs associated with FPGA emulation and the necessary RTL implementation of the proposed design.

### B. Accelerator Design Space Exploration

Since the early days, many works have focused on exploring optimal hardware designs as well as mapping [14], [15], [27], [36], [37], [51], [60], [74], [81], [84]. LLMCompass is different from these works in design considerations and emphasis: (i) Mainly targeting Convolutional Neural Networks (CNNs), these works focus on loop parallelization, loop order, and data flows (e.g., weight stationary or output stationary), which are not the primary design considerations in Transformer-based LLMs. LLMCompass is more tailored for matrix multiplication tiling and scheduling as well as other Transformer operators such as LayerNorm. (ii) LLMCompass is designed for GPU-scale designs, which are much larger than CNN accelerators like Eyeriss [10]. LLM workloads are also significantly larger than CNN workloads.

LLMCompass can also complement design space explorations. Implemented as a Python library, LLMCompass can be seamlessly integrated into design space exploration frameworks such as FAST [84]. FAST uses an internal TPU performance simulator, limiting its broader utility. Fast and accurate, we believe the fully open-source LLMCompass can democratize hardware design space exploration research.

### C. Accelerating LLM Inference

Many Transformer accelerators have been proposed [26], [67], [75], [76], mainly focusing on accelerating the Transformer with hardware-software co-design such as pruning or approximate-computing. Whether these techniques are effective for the largest of models remains to be seen. Additionally, the major challenge of LLMs today comes from the massive scale of the models, which is the main scope of this paper.

Many efforts have also been made to accelerate LLM inference at the software domain [3], [5], [12], [13], [19], [54], [56], [63], [82]. LLMCompass is compatible with these optimization techniques by modeling their compute and memory access patterns, as discussed in Section VI-2.

---

TABLE V: Comparison of Hardware Evaluation Methods

| Methods       | Fast Accurate | Architecturally Descriptive ✧ | Performance Cost-Optimal Aware |
|---------------|---------------|---------------------------------|---------------------------------|
| Roofline      | ✗             | ✗                               | ✗                               |
| Cycle-level   | ✓             | ✓                               | ✗                               |
| FPGA          | ✓             | ✗                               | ✓                               |
| LLMCompass    | ✓             | ✓                               | ✓                               |

✧: The ability to describe different hardware designs.

---

### VIII. Conclusion

This work introduces LLMCompass, a fast, accurate, and architecturally descriptive hardware evaluation framework for LLM inference workloads. LLMCompass’ hardware description template, mapper, and architectural simulator allow hardware designers to evaluate large-scale chip designs for LLMs, which are infeasible for cycle-level simulators. The incorporated area and cost models can also help designers reason about performance-cost trade-offs. With the aid of LLMCompass, we draw new insights on how hardware designs should be tailored for LLM inference workloads. We propose a latency-oriented design and a throughput-oriented design that achieve 1.06× and 3.41× performance per cost improvements respectively, compared to NVIDIA GA100. We plan to extend LLMCompass to support more machine learning workloads as well as LLM training/fine-tuning in the future.

---

### Acknowledgements

We would like to thank Qixuan (Max) Yu, Zhongning Yu, Haiyue Ma, Yanghui Qu, Christopher Batten, and the entire Princeton Parallel Group, for their feedback, suggestions, and encouragement. This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-2039656, the National Science Foundation under Grant No. CCF-1822949, Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) under agreement No. FA8650-18-2-7862. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) or the U.S. Government.
