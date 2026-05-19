# Part 4: Tensors — The Generalization That Powers Deep Learning

---

## The Problem Tensors Solve

We have:
- **Lists** → flexible but slow
- **NumPy arrays** → fast on CPU, any dimension
- **Matrices** → the math concept (2D only)

But for deep learning we need:
1. Multi-dimensional data (images, video, batches) → **NumPy can do this**
2. GPU acceleration (1000x faster) → **NumPy CANNOT do this**
3. Automatic gradient computation → **NumPy CANNOT do this**

**Tensors = NumPy arrays + GPU + Autograd**

---

## What is a Tensor?

A tensor is a **multi-dimensional array of numbers** that:
- Lives on CPU **or** GPU
- Can track every operation done to it
- Can automatically compute derivatives (gradients)

```python
import torch

# This looks like NumPy...
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# ...but it can do things NumPy can't:
x = x.to('cuda')              # move to GPU
x.requires_grad = True         # track operations for gradients
```

---

## Tensor vs NumPy Array — Side by Side

```python
import numpy as np
import torch

# Creation — almost identical syntax
np_arr = np.array([1.0, 2.0, 3.0])
tensor = torch.tensor([1.0, 2.0, 3.0])

# Math — identical
np_result = np_arr * 2 + 1
torch_result = tensor * 2 + 1

# Shape — identical
np_arr.shape      # (3,)
tensor.shape      # torch.Size([3])

# Reshape — identical
np_arr.reshape(1, 3)
tensor.reshape(1, 3)
```

So far, same thing. Here's where tensors diverge:

```python
# ═══════════════════════════════════════════
# DIFFERENCE 1: GPU
# ═══════════════════════════════════════════
# NumPy: stuck on CPU
np_big = np.random.randn(10000, 10000)
# np_big @ np_big  → takes ~2 seconds on CPU

# Tensor: can use GPU
torch_big = torch.randn(10000, 10000, device='cuda')
# torch_big @ torch_big  → takes ~5 milliseconds on GPU (400x faster!)


# ═══════════════════════════════════════════
# DIFFERENCE 2: Automatic Gradients
# ═══════════════════════════════════════════
# NumPy: you compute derivatives by hand or with finite differences
# No built-in way to get dy/dx

# Tensor: automatic!
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 5*x + 3    # y = x² + 5x + 3
y.backward()            # compute dy/dx automatically
print(x.grad)           # 11.0 (because dy/dx = 2x + 5 = 11 at x=3)


# ═══════════════════════════════════════════
# DIFFERENCE 3: Computation Graph
# ═══════════════════════════════════════════
# Every operation is recorded, forming a graph:
# x → x² → +5x → +3 → y
# This graph is walked BACKWARD to compute gradients
# This is how neural networks learn!
```

---

## Where Do We Use Tensors?

### 1. Deep Learning (The Primary Use Case)

```python
# Every neural network layer is a tensor operation:
input = torch.randn(64, 784)       # 64 images, flattened
weights = torch.randn(784, 128)    # learned transformation
output = input @ weights            # matrix multiply → (64, 128)

# The ENTIRE training process:
# Forward:  input tensors → through weight tensors → output tensor
# Loss:     compare output tensor to target tensor → loss (scalar tensor)
# Backward: compute gradient tensors for all weight tensors
# Update:   modify weight tensors using gradient tensors
```

### 2. Computer Vision

```python
# A batch of images
images = torch.randn(32, 3, 224, 224)
# 32 images, 3 color channels (RGB), 224×224 pixels

# Convolution (pattern detection) — tensor operation
conv = torch.nn.Conv2d(3, 64, kernel_size=3)
features = conv(images)  # (32, 64, 222, 222) — 64 feature maps
```

### 3. Natural Language Processing

```python
# A batch of sentences
# Each word → integer index → embedding vector
token_ids = torch.randint(0, 50000, (16, 128))  # 16 sentences, 128 tokens
embedding = torch.nn.Embedding(50000, 768)
vectors = embedding(token_ids)  # (16, 128, 768) — each token is now a 768-dim vector
```

### 4. Audio Processing

```python
# Spectrogram of audio
audio = torch.randn(8, 1, 128, 200)  # 8 clips, 1 channel, 128 mel bins, 200 time steps
```

### 5. Reinforcement Learning

```python
# Game state as tensor
state = torch.tensor([0.5, -0.3, 1.2, 0.0])  # position, velocity, angle, etc.
# Policy network outputs action probabilities
action_probs = policy_net(state)  # tensor of probabilities
```

### 6. Scientific Simulation (on GPU)

```python
# Fluid simulation: 3D grid of velocities
velocity_field = torch.randn(100, 100, 100, 3, device='cuda')
# 100×100×100 grid, 3 velocity components (x, y, z)
# GPU makes this 100x faster than NumPy
```

---

## Why Tensors Need GPU

### The Math of Deep Learning is MASSIVE

```
A simple image classifier:
  Input:  (64, 3, 224, 224) = 9,633,792 numbers
  Layer 1 weights: (3, 64, 3, 3) = 1,728 numbers
  ...
  Total operations per forward pass: ~4 BILLION multiply-adds

Training = doing this thousands of times.
  CPU: 4 billion ops × 1000 batches = hours
  GPU: same thing = minutes
```

### Why GPU is Faster

```
CPU:  8 cores, each very fast, doing things one at a time
GPU:  5000+ cores, each slower, but ALL working simultaneously

Matrix multiply (1024 × 1024):
  CPU: 8 cores × sequential chunks = slow
  GPU: 5000 cores × all elements at once = fast

It's like:
  CPU = 8 expert chefs cooking one dish at a time
  GPU = 5000 line cooks each handling one ingredient simultaneously
```

---

## Why Tensors Need Autograd

### The Problem: How Does a Network Learn?

```
Network has 10 million weight values.
Each weight needs to be adjusted to reduce the error.
To know WHICH DIRECTION to adjust, you need the GRADIENT (derivative).

Computing 10 million derivatives by hand? Impossible.
Computing them with finite differences? Too slow (need 10M extra forward passes).

Autograd: records all operations, walks backward through them,
          computes ALL 10 million gradients in ONE backward pass.
```

### How It Works (Simplified)

```python
# Forward pass — PyTorch secretly records everything
x = torch.tensor(2.0, requires_grad=True)
a = x * 3        # recorded: a = 3x
b = a + 1        # recorded: b = a + 1
c = b ** 2       # recorded: c = b²

# Backward pass — walks the recording in reverse
c.backward()
# dc/db = 2b = 2(3x+1) = 2(7) = 14
# db/da = 1
# da/dx = 3
# dc/dx = dc/db × db/da × da/dx = 14 × 1 × 3 = 42

print(x.grad)  # 42.0 ✓
```

This is the **chain rule** applied automatically. No manual math needed.

---

## What's Actually Inside a Tensor (Programmatically)

A PyTorch tensor is a Python wrapper around a C++ object (`at::Tensor`). Here's what it holds:

```
┌─────────────────────────────────────────────────┐
│  torch.Tensor (Python wrapper)                  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Storage (raw data)                       │  │
│  │  - data_ptr → contiguous block of memory  │  │
│  │  - dtype (float32, int64, etc.)           │  │
│  │  - device (CPU/CUDA)                      │  │
│  │  - nbytes                                 │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Metadata (view into storage)             │  │
│  │  - shape/size                             │  │
│  │  - stride (step size per dimension)       │  │
│  │  - offset                                 │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Autograd Metadata (if requires_grad=True)│  │
│  │  - grad_fn → pointer to backward function │  │
│  │  - grad → accumulated gradient tensor     │  │
│  │  - requires_grad (bool flag)              │  │
│  │  - is_leaf (bool)                         │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Why It's Faster Than a Python List (Basic Array)

```
Python List [1.0, 2.0, 3.0]:
  → 3 separate Python float objects scattered on the heap
  → Each has reference count, type pointer, value
  → ~28 bytes PER float object + 8 bytes pointer in list
  → Total: ~108 bytes for 3 numbers
  → To multiply: Python loop, unbox each float, multiply, rebox, store

Tensor [1.0, 2.0, 3.0] (float32):
  → 12 contiguous bytes (3 × 4 bytes), nothing else
  → One C++ kernel call processes ALL elements
  → No Python loop, no boxing/unboxing
  → Total: 12 bytes for 3 numbers
```

### The grad_fn Chain (How the Computation Graph is Built)

When `requires_grad=True`, each operation creates a `grad_fn` node that forms a linked graph:

```python
x = torch.tensor(3.0, requires_grad=True)  # leaf, grad_fn = None
t1 = x ** 2        # t1.grad_fn = PowBackward0 → stores "parent=x, exponent=2"
t2 = 2 * x         # t2.grad_fn = MulBackward0 → stores "parent=x, multiplier=2"
y = t1 + t2 + 1    # y.grad_fn = AddBackward0 → points to t1 and t2's grad_fns
```

Each `grad_fn` stores:
- Pointer to **parent node(s)** (forming a DAG — Directed Acyclic Graph)
- **Saved values** needed to compute the derivative (e.g., exponent `2` for PowBackward)

When you call `y.backward()`, it walks this chain in reverse, applying the chain rule at each node.

### Inspect It Yourself

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1

print(x.storage())                # raw memory: [3.0]
print(x.stride())                 # (1,) — step size
print(x.data_ptr())               # actual memory address (integer)
print(y.grad_fn)                  # AddBackward0
print(y.grad_fn.next_functions)   # parent nodes in the graph
```

---

## Contiguous vs Non-Contiguous Memory

This is about **how elements are physically laid out in memory** vs how you logically see them.

### Contiguous = Elements Stored in Order, No Gaps

```
Logical view (2×3 matrix):
[[1, 2, 3],
 [4, 5, 6]]

Physical memory (row-major / C-contiguous):
[1, 2, 3, 4, 5, 6]  ← one continuous block, row after row

Stride = (3, 1)
  → Next row: jump 3 positions in memory
  → Next column: jump 1 position in memory
```

### Non-Contiguous = Logical Order ≠ Physical Order

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = x.t()  # transpose — NO data is copied!
```

After transpose, you logically see:
```
[[1, 4],
 [2, 5],
 [3, 6]]
```

But physical memory is still `[1, 2, 3, 4, 5, 6]`. PyTorch just changed the strides:

```
Original x:  stride = (3, 1)  → next row: +3, next col: +1
Transposed:  stride = (1, 3)  → next row: +1, next col: +3
```

The elements you read "in order" are now scattered in memory — that's non-contiguous.

### Visual Summary

```
CONTIGUOUS (stride matches shape):
Memory:  [a][b][c][d][e][f]
Reading: [a][b][c][d][e][f]  ← sequential, cache-friendly ✓

NON-CONTIGUOUS (stride jumps around):
Memory:  [a][b][c][d][e][f]
Reading: [a]      [d]         ← row 0
            [b]      [e]      ← row 1
               [c]      [f]   ← row 2
                              ← jumping around, cache-unfriendly ✗
```

### Why Does It Matter?

1. **Speed** — CPUs/GPUs load memory in chunks (cache lines). Sequential access = fast. Jumping around = cache misses = slow.

2. **Some operations require contiguous data** — they'll error out or silently call `.contiguous()` which copies data into a fresh contiguous block.

### Operations That Make Tensors Non-Contiguous

```python
x.t()              # transpose
x.permute(2, 0, 1) # reorder dimensions
x.narrow(0, 1, 2)  # slice a subset
x[::2]             # step slicing
```

All of these reuse the same memory with different strides — no copy, but non-contiguous.

### How to Fix It

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = x.t()
print(y.is_contiguous())    # False
print(y.stride())           # (1, 3) — jumping around

z = y.contiguous()          # copies into fresh contiguous memory
print(z.is_contiguous())    # True
print(z.stride())           # (2, 1) — proper layout for 3×2 matrix
```

### The Key Insight: Views vs Copies

```
View (non-contiguous):  Same memory, different stride → FREE but scattered
Copy (contiguous):      New memory, proper layout → COSTS memory but fast access

PyTorch prefers views (no copy) for efficiency.
You only pay the copy cost when you actually need contiguous data.
```

---

## The Evolution — Complete Picture

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Python List                                                     │
│  └── General container, any type, slow for math                  │
│                                                                  │
│       ↓ "I need fast math on numbers"                            │
│                                                                  │
│  NumPy Array                                                     │
│  └── Fixed type, contiguous memory, vectorized C ops, CPU only   │
│                                                                  │
│       ↓ "I need GPU speed and automatic gradients"               │
│                                                                  │
│  PyTorch Tensor                                                  │
│  └── Everything NumPy has + GPU + Autograd + Computation Graph   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### When to Use What?

```
Use Python Lists when:
  → Small data, mixed types, general programming
  → Building up data dynamically
  → Non-numerical work

Use NumPy Arrays when:
  → Data analysis, statistics, signal processing
  → Scientific computing (no GPU needed)
  → Working with pandas, matplotlib, scipy
  → CPU is fast enough for your problem

Use PyTorch Tensors when:
  → Deep learning (training neural networks)
  → You need GPU acceleration
  → You need automatic differentiation
  → Large-scale matrix computations
  → Anything where NumPy is too slow
```

---

## Summary

```
PyTorch Tensor:
  ✅ Everything NumPy can do (and same syntax)
  ✅ GPU acceleration (100-1000x faster for large math)
  ✅ Automatic differentiation (gradients for free)
  ✅ Computation graph (enables backpropagation)
  ✅ Seamless NumPy interop (.numpy(), from_numpy())
  ✅ Ecosystem (torchvision, torchaudio, transformers)
  
  ❌ Slightly more complex API than NumPy
  ❌ Requires NVIDIA GPU for CUDA acceleration
  ❌ Overkill for simple data analysis (use NumPy/pandas instead)
```

---

## The Complete Story

```
Part 1: Lists      → "I can hold anything, but I'm slow at math"
Part 2: NumPy      → "I'm fast at math on CPU, with shape awareness"
Part 3: Matrices   → "I'm the mathematical concept — 2D grids that transform data"
Part 4: Tensors    → "I'm NumPy + GPU + Autograd — I power deep learning"
```

**Next → Part 5: Comparison Table & When to Use What (Cheat Sheet)**
