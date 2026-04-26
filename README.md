# ML Parallelism Workshop

This workshop teaches you how to train a convolutional neural network on the CIFAR-10 dataset and then scale that training across multiple GPUs using three different parallelism strategies. You will start from a single-GPU baseline and work through data parallelism, pipeline parallelism, and tensor parallelism, seeing how each approach distributes work differently and where each one pays off.

All notebooks run on Kaggle using a free session with two T4 GPUs.

---

## Workshop Goals

By the end of this workshop, you will be able to:

- Train a CNN from scratch using standard PyTorch patterns
- Replicate the model across GPUs with Distributed Data Parallel (DDP) and synchronize gradients automatically
- Split a model across GPUs by stage using pipeline parallelism and overlap computation with microbatching
- Shard individual weight matrices across GPUs using tensor parallelism
- Reason about the tradeoffs between each strategy: communication overhead, memory savings, and when each makes sense

---

## The Dataset: CIFAR-10

CIFAR-10 contains 60,000 color images at 32x32 resolution, split into 50,000 training images and 10,000 test images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

It is small enough to train quickly on free GPU hardware, but large enough that parallelism produces measurable speedups.

---

## The Model: CIFARNet

CIFARNet is a custom CNN defined in each notebook. It has two main sections:

**Convolutional layers** (feature extraction):
- Conv1: 3 -> 64 channels
- Conv2: 64 -> 128 channels
- Conv3: 128 -> 256 channels
- Conv4: 256 -> 512 channels

Each conv layer uses batch normalization, ReLU activation, and max pooling to downsample spatial dimensions.

**Fully connected layers** (classification):
- FC1: 32,768 -> 512
- FC2: 512 -> 256
- FC3: 256 -> 10 (one output per class)

FC1 and FC2 include 50% dropout to prevent overfitting. Training uses SGD with momentum, cosine annealing for the learning rate, and random horizontal flips and crops for data augmentation.

---

## Parallelism Strategies

### Single GPU (Baseline)

The model and all data live on one GPU. This is the reference point you will use to measure speedups in the later notebooks.

### Data Parallel (DDP)

The full model is copied to each GPU. Each GPU processes a different slice of each mini-batch, computes gradients independently, and then the gradients are averaged across all GPUs before the weights are updated. The model stays identical on all GPUs at all times. This approach scales well when the model fits on a single GPU and the bottleneck is throughput.

### Pipeline Parallel

The model is split into sequential stages, each living on a different GPU. A batch is divided into smaller microbatches that flow through the pipeline one stage at a time. While GPU 1 is processing microbatch 2, GPU 0 is already working on microbatch 3, overlapping computation to reduce idle time. This approach is useful when the model is too large to fit on a single device.

### Tensor Parallel

Individual weight matrices inside the model are sharded across GPUs. For a matrix multiply, one GPU computes part of the output columns and another computes the rest. An all-reduce operation combines the partial results. Unlike DDP, each GPU holds only a fraction of each sharded layer's parameters. This is the most fine-grained form of parallelism and is typically combined with the other strategies for very large models.

---

## Notebooks

The workshop is structured as a progression. Each notebook builds on the concepts from the previous one.

| # | Notebook | Topic | Key Concepts |
|---|----------|-------|--------------|
| 1 | `01_single_gpu.ipynb` | Baseline training | `nn.Module`, training loop, SGD, CosineAnnealingLR, saving weights |
| 2 | `02_data_parallel.ipynb` | Distributed Data Parallel | `DistributedDataParallel`, `DistributedSampler`, `mp.spawn`, NCCL process group |
| 3 | `03_pipeline_parallel.ipynb` | Pipeline parallelism | `StagedCIFARNet`, `PipelineStage`, `ScheduleGPipe`, microbatching |
| 4 | `04_tensor_parallel.ipynb` | Tensor parallelism | `DeviceMesh`, `parallelize_module`, `ColwiseParallel`, `RowwiseParallel`, DTensor |

Each notebook is fully self-contained with no external imports. Distributed notebooks use `torch.multiprocessing.spawn` so they work directly from notebook cells without a separate launcher.

Two versions of each notebook are provided:

- `student_versions/` contains fill-in-the-blank sections marked with `### YOUR CODE HERE ###`
- `solutions/` contains the complete implementations

---

## Getting Started on Kaggle

1. Open the notebook you want to run on Kaggle (or upload it from this repository).
2. Go to **Settings** on the right sidebar and set the Accelerator to **GPU T4 x2**.
3. Run the cells in order from top to bottom.

For the student versions, fill in the marked sections before running. The solution notebooks are available if you get stuck.

---

## Additional Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Pipeline Parallelism](https://pytorch.org/docs/stable/distributed.pipelining.html)
- [PyTorch Tensor Parallelism](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
