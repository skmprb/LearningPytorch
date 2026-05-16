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
