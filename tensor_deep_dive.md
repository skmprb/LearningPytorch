# Understanding Tensors — Complete Deep Dive

---

## Section 1: What is a Tensor? (The Big Picture)

### In Plain English

A tensor is just a **container for numbers** arranged in a grid of any number of dimensions.

You already know tensors — you just didn't call them that:

```
A single number         → Tensor (0 dimensions)
A list of numbers       → Tensor (1 dimension)
A table of numbers      → Tensor (2 dimensions)
A cube of numbers       → Tensor (3 dimensions)
A stack of cubes        → Tensor (4 dimensions)
...and so on
```

### Building Up Dimension by Dimension

#### 0D — Scalar (a single number)

```
5
```

Just one value. Temperature today. Your age. A single pixel brightness.

```python
import torch
x = torch.tensor(5)
# shape: ()  — no dimensions
# ndim: 0
```

#### 1D — Vector (a list of numbers)

```
[1, 2, 3, 4, 5]
```

A row of values. Audio signal at 5 time steps. Embedding of a word. Features of one data point.

```python
x = torch.tensor([1, 2, 3, 4, 5])
# shape: (5,)  — one dimension of size 5
# ndim: 1
```

#### 2D — Matrix (a table of numbers)

```
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```

Rows and columns. A grayscale image. A spreadsheet. A batch of feature vectors.

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
# shape: (3, 3)  — 3 rows, 3 columns
# ndim: 2
```

#### 3D — A cube (or a stack of matrices)

```
Imagine stacking multiple tables on top of each other:

Layer 0:        Layer 1:        Layer 2:
[[1, 2, 3],    [[10, 11, 12],  [[19, 20, 21],
 [4, 5, 6],     [13, 14, 15],   [22, 23, 24],
 [7, 8, 9]]     [16, 17, 18]]   [25, 26, 27]]
```

A color image (3 layers = R, G, B). A batch of sequences. Time series with multiple features.

```python
x = torch.randn(3, 3, 3)
# shape: (3, 3, 3)  — 3 layers, each 3×3
# ndim: 3
```

#### 4D — A stack of cubes

```
Multiple color images stacked together (a batch):

Image 0:  (3, 224, 224)  — RGB, height, width
Image 1:  (3, 224, 224)
Image 2:  (3, 224, 224)
...
Image 31: (3, 224, 224)

Together: (32, 3, 224, 224) — batch of 32 images
```

```python
batch = torch.randn(32, 3, 224, 224)
# shape: (32, 3, 224, 224)
# ndim: 4
```

#### 5D — A sequence of batches (video)

```
A batch of videos:
(batch, time_frames, channels, height, width)
(8, 30, 3, 224, 224) — 8 videos, 30 frames each, RGB, 224×224
```

### Real-World Analogies

| Dimensions | Tensor | Real-World Example |
|-----------|--------|-------------------|
| 0D | Scalar | Temperature: `72` |
| 1D | Vector | Audio sample: `[0.2, -0.1, 0.5, ...]` |
| 2D | Matrix | Grayscale photo: 28×28 pixels |
| 3D | 3D Tensor | Color photo: 3×224×224 (RGB) |
| 4D | 4D Tensor | Batch of photos: 32×3×224×224 |
| 5D | 5D Tensor | Batch of videos: 8×30×3×224×224 |

### Why the Name "Tensor"?

The word comes from **physics and mathematics** (Latin: *tendere* = "to stretch").

In physics, tensors describe things that have magnitude AND direction in multiple dimensions — like stress in a material, or the curvature of spacetime.

In deep learning, we borrowed the name because:
- Our data is multi-dimensional
- We need a general word for "n-dimensional array"
- "Tensor" sounds cooler than "multi-dimensional array" 😄

**The key insight**: A tensor is just a generalization. A scalar is a 0D tensor. A vector is a 1D tensor. A matrix is a 2D tensor. Beyond that, we just say "tensor."

### The One-Line Definition

> **A tensor is an n-dimensional array of numbers with a fixed shape, a data type, and a device (CPU/GPU).**

---

### ⚠️ Common Mistake

**Confusing "tensor" with "3D+ array"**

Many people think tensors must be 3D or higher. Wrong!
- A scalar IS a tensor (0D)
- A vector IS a tensor (1D)
- A matrix IS a tensor (2D)

"Tensor" is the umbrella term for ALL of these.

---


## Section 2: Tensor Dimensions & Shapes

### In Plain English

Every tensor has a **shape** — it tells you how many numbers are in each direction.

Think of shape as the **blueprint** of the container:
- `(5,)` → a row with 5 slots
- `(3, 4)` → a table with 3 rows and 4 columns
- `(2, 3, 4)` → 2 tables, each with 3 rows and 4 columns

### Key Vocabulary

| Term | Meaning | Example |
|------|---------|---------|
| **Rank** (ndim) | Number of dimensions | Matrix has rank 2 |
| **Shape** | Size along each dimension | `(3, 4)` = 3 rows, 4 cols |
| **Axis** (dim) | A specific dimension | Axis 0 = rows, Axis 1 = cols |
| **Size** | Total number of elements | `(3, 4)` → 12 elements |

### Understanding Axes

Axes are **numbered from 0** (outermost) to n-1 (innermost).

```
Shape: (batch, channels, height, width)
         ↓       ↓        ↓       ↓
       Axis 0  Axis 1   Axis 2  Axis 3
```

#### 1D Tensor — One Axis

```
Axis 0 →
[10, 20, 30, 40, 50]

Shape: (5,)
Axis 0 has size 5
```

#### 2D Tensor — Two Axes

```
         Axis 1 →
        col0  col1  col2  col3
Axis 0  [ 1,    2,    3,    4]    ← row 0
  ↓     [ 5,    6,    7,    8]    ← row 1
        [ 9,   10,   11,   12]    ← row 2

Shape: (3, 4)
Axis 0 (rows) has size 3
Axis 1 (cols) has size 4
```

#### 3D Tensor — Three Axes

```
Axis 0 = which "layer" (depth)
Axis 1 = which row
Axis 2 = which column

        Layer 0 (Axis 0 = 0)         Layer 1 (Axis 0 = 1)
        ┌─────────────────┐          ┌─────────────────┐
        │  1   2   3   4  │          │ 13  14  15  16  │
        │  5   6   7   8  │          │ 17  18  19  20  │
        │  9  10  11  12  │          │ 21  22  23  24  │
        └─────────────────┘          └─────────────────┘

Shape: (2, 3, 4)
Axis 0 (layers) = 2
Axis 1 (rows)   = 3
Axis 2 (cols)   = 4
Total elements  = 2 × 3 × 4 = 24
```

### Reading Shapes — The Skill You Need

When you see a shape, read it **left to right** as "how many of what":

```
(32, 3, 224, 224)
  │   │    │    │
  │   │    │    └── 224 pixels wide
  │   │    └─────── 224 pixels tall
  │   └──────────── 3 color channels (RGB)
  └──────────────── 32 images in the batch
```

More examples:

```
(1000, 10)        → 1000 samples, each with 10 features
(64, 50, 128)     → 64 sentences, each 50 words, each word is 128-dim embedding
(8, 30, 3, 64, 64) → 8 videos, 30 frames, RGB, 64×64 pixels
```

### Negative Indexing for Axes

Just like Python lists, you can count from the end:

```
Shape: (32, 3, 224, 224)

Axis  0 =  Axis -4  → batch
Axis  1 =  Axis -3  → channels
Axis  2 =  Axis -2  → height
Axis  3 =  Axis -1  → width (last dimension)
```

This is useful because the **last dimension** is always `dim=-1` regardless of how many dimensions the tensor has.

### Shape Arithmetic

```python
# Total elements = product of all dimensions
torch.tensor(shape).prod()

# Examples:
# (3, 4)       → 12 elements
# (2, 3, 4)    → 24 elements
# (32, 3, 224, 224) → 32 × 3 × 224 × 224 = 4,816,896 elements
```

### The "Batch Dimension" Convention

In deep learning, **Axis 0 is almost always the batch dimension**.

```
Single image:  (3, 224, 224)        ← no batch
Batch:         (32, 3, 224, 224)    ← 32 images

Single sentence: (50, 128)          ← 50 tokens, 128-dim
Batch:           (64, 50, 128)      ← 64 sentences
```

Why? Because we process multiple samples at once for efficiency (parallelism on GPU).

---

### ⚠️ Common Mistake

**Confusing axis order**

```
Image in PyTorch:  (C, H, W)     → channels FIRST
Image in NumPy:    (H, W, C)     → channels LAST

# This WILL break your code if you mix them up!
# Use .permute() to convert:
# numpy_style (H, W, C) → pytorch_style (C, H, W)
# tensor.permute(2, 0, 1)
```

---


## Section 3: Why Tensors? (Why Not Just NumPy Arrays?)

### In Plain English

NumPy arrays can do everything tensors do... on CPU. So why did PyTorch invent its own?

Because tensors have **3 superpowers** that NumPy arrays don't:

### Superpower 1: GPU Acceleration

```
NumPy array  → lives on CPU only
PyTorch tensor → can live on CPU OR GPU

GPU = thousands of cores doing math in parallel
CPU = a few powerful cores doing math sequentially
```

For a matrix multiplication of size (1000, 1000) × (1000, 1000):
- CPU: ~100 milliseconds
- GPU: ~1 millisecond (100x faster!)

```python
# Move to GPU with one line
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
z = x @ y  # happens on GPU — massively parallel
```

### Superpower 2: Automatic Differentiation (Autograd)

This is THE reason deep learning works.

```
NumPy:    you compute gradients manually (painful, error-prone)
PyTorch:  tensors REMEMBER every operation done to them
          and can automatically compute gradients (derivatives)
```

```python
x = torch.tensor(3.0, requires_grad=True)  # "track this!"
y = x ** 2 + 2 * x                         # PyTorch records: y = x² + 2x
y.backward()                                 # compute dy/dx automatically
print(x.grad)                                # 8.0 (because 2x + 2 = 8 at x=3)
```

Without this, training neural networks would be nearly impossible.

### Superpower 3: Broadcasting

Operate on tensors of **different shapes** without copying data:

```python
# Add a scalar to every element (no loop needed!)
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])   # shape (2, 3)
y = x + 10                      # 10 "broadcasts" to shape (2, 3)
# Result: [[11, 12, 13], [14, 15, 16]]

# Subtract mean from each column
data = torch.randn(100, 5)      # 100 samples, 5 features
mean = data.mean(dim=0)         # shape (5,)
centered = data - mean          # (100, 5) - (5,) → broadcasts!
```

### How Neural Networks See Data as Tensors

Everything in deep learning is a tensor operation:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Input Tensor  ──→  Weight Tensor  ──→  Output Tensor │
│   (your data)        (learned)           (prediction)  │
│                                                         │
│   (64, 784)     ×    (784, 128)     =    (64, 128)    │
│   64 images          learned params       64 outputs   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

That's it. Deep learning = tensor in → multiply by weight tensors → tensor out.

### Common Tensor Shapes in Deep Learning

#### Images (Computer Vision)

```
Single image:     (C, H, W)           → (3, 224, 224)
Batch of images:  (B, C, H, W)        → (32, 3, 224, 224)

B = batch size
C = channels (3 for RGB, 1 for grayscale)
H = height in pixels
W = width in pixels
```

#### Text (NLP)

```
Single sentence:    (seq_len, embed_dim)      → (50, 768)
Batch of sentences: (B, seq_len, embed_dim)   → (32, 50, 768)

B = batch size
seq_len = number of tokens (words/subwords)
embed_dim = size of each token's vector representation
```

#### Tabular Data

```
Single sample:    (num_features,)        → (10,)
Batch:            (B, num_features)      → (64, 10)
```

#### Audio

```
Single waveform:  (num_samples,)              → (16000,)  — 1 second at 16kHz
Spectrogram:      (n_mels, time_steps)        → (128, 100)
Batch:            (B, n_mels, time_steps)     → (32, 128, 100)
```

#### Video

```
Single video:     (T, C, H, W)           → (30, 3, 224, 224)
Batch:            (B, T, C, H, W)        → (4, 30, 3, 224, 224)

T = number of frames
```

### The Batch Dimension — Why?

Every tensor in training has an extra first dimension: the **batch**.

Why not process one sample at a time?

```
1 sample at a time:
  - GPU sits mostly idle (not enough work to parallelize)
  - Very slow training

64 samples at a time (batch):
  - GPU cores all busy
  - 64x more efficient
  - Gradients are averaged → smoother training
```

Rule: **If your tensor doesn't have a batch dimension, something is wrong.**

---

### ⚠️ Common Mistake

**Forgetting the batch dimension**

```python
# ❌ WRONG: passing a single image without batch dim
image = torch.randn(3, 224, 224)       # shape (3, 224, 224)
output = model(image)                   # ERROR! Model expects (B, 3, 224, 224)

# ✅ CORRECT: add batch dimension
image = image.unsqueeze(0)             # shape (1, 3, 224, 224)
output = model(image)                   # works!
```

---


## Section 4: Tensor Memory Layout

### In Plain English

Your computer's memory is **flat** — just a long line of slots. But tensors are multi-dimensional. So how does a 2D table fit into a 1D line of memory?

Answer: **strides** — a recipe for jumping through flat memory to find the right element.

### How a 2D Tensor Lives in Memory

```
Tensor (logical view):          Memory (physical, flat):
┌─────────────┐
│  1   2   3  │                 [1, 2, 3, 4, 5, 6, 7, 8, 9]
│  4   5   6  │                  ↑        ↑        ↑
│  7   8   9  │                 row 0    row 1    row 2
└─────────────┘

Shape: (3, 3)
Stored row by row (row-major order)
```

To find element `[row, col]` in flat memory:
```
position = row × 3 + col × 1

[0, 0] → 0×3 + 0×1 = 0  → memory[0] = 1
[1, 2] → 1×3 + 2×1 = 5  → memory[5] = 6
[2, 1] → 2×3 + 1×1 = 7  → memory[7] = 8
```

Those multipliers (3 and 1) are the **strides**.

### What Are Strides?

**Stride** = how many memory positions to skip to move one step along each axis.

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(x.stride())  # (3, 1)
```

Reading `stride = (3, 1)`:
- To move one step along **axis 0** (next row): skip **3** elements
- To move one step along **axis 1** (next col): skip **1** element

### Strides for Higher Dimensions

```python
x = torch.randn(2, 3, 4)
print(x.stride())  # (12, 4, 1)

# To move to next "layer" (axis 0): skip 12 elements (3×4)
# To move to next row (axis 1):     skip 4 elements
# To move to next col (axis 2):     skip 1 element
```

General rule: `stride[i] = product of all dimensions after i`

```
Shape:   (2,  3,  4)
Strides: (12, 4,  1)
          │   │   └── 1 (innermost always 1)
          │   └────── 4 (= 4)
          └────────── 12 (= 3 × 4)
```

### Why Strides Matter: Views vs Copies

#### View (No Copy — Shares Memory)

When you reshape a tensor with `.view()` or `.transpose()`, PyTorch doesn't copy data. It just changes the strides.

```python
x = torch.arange(12)          # [0,1,2,3,4,5,6,7,8,9,10,11]
                                # shape (12,), stride (1,)

y = x.view(3, 4)              # Same memory, different strides!
                                # shape (3, 4), stride (4, 1)

# Proof: modifying y changes x
y[0, 0] = 99
print(x[0])  # 99 — same memory!
```

#### Copy (New Memory)

When you call `.clone()` or `.contiguous()` on a non-contiguous tensor, PyTorch allocates new memory.

```python
z = y.clone()    # new memory — independent copy
z[0, 0] = 0
print(y[0, 0])  # still 99 — z is independent
```

### Contiguous vs Non-Contiguous

A tensor is **contiguous** if its elements are laid out in memory in the expected order (row by row, no gaps, no weird jumps).

```python
x = torch.arange(12).view(3, 4)
print(x.is_contiguous())  # True — normal layout

# Transpose changes strides but NOT memory layout
y = x.t()  # transpose
print(y.is_contiguous())  # False! — strides are "backwards"
print(y.stride())         # (1, 4) instead of (4, 1)
```

Why does this matter?

```python
# .view() requires contiguous memory
# y.view(12)  ← ERROR! y is not contiguous

# Fix: make it contiguous first (copies data)
z = y.contiguous()
z.view(12)  # works now

# Or use .reshape() which handles this automatically
y.reshape(12)  # works — makes contiguous copy if needed
```

### View vs Reshape — The Rule

```
.view()    → FAST (no copy), but FAILS if not contiguous
.reshape() → ALWAYS works (copies if needed, view if possible)
```

Use `.reshape()` when you don't care about memory sharing.
Use `.view()` when you explicitly want to share memory (and know it's contiguous).

### Visual Summary

```
┌──────────────────────────────────────────────────────────┐
│                    FLAT MEMORY                            │
│  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]              │
└──────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   view(3, 4)           view(4, 3)           view(2, 6)
   stride=(4,1)         stride=(3,1)         stride=(6,1)
   ┌─────────┐          ┌───────┐           ┌───────────┐
   │ 1  2  3  4│        │ 1  2  3│          │ 1  2  3  4  5  6│
   │ 5  6  7  8│        │ 4  5  6│          │ 7  8  9 10 11 12│
   │ 9 10 11 12│        │ 7  8  9│          └───────────┘
   └─────────┘          │10 11 12│
                         └───────┘

Same memory, different shapes — just different strides!
```

---

### ⚠️ Common Mistake

**Assuming reshape always creates a new tensor**

```python
x = torch.arange(12)
y = x.view(3, 4)

# y is NOT a copy — it's a VIEW of x
y[0, 0] = 999
print(x[0])  # 999! x was modified too!

# If you need independence, explicitly clone:
y = x.view(3, 4).clone()
```

---


## Section 5: Tensor Operations — The Mental Model

### In Plain English

There are only **5 categories** of tensor operations. Once you understand these, you understand ALL of PyTorch:

1. **Element-wise** — same shape in, same shape out
2. **Reduction** — collapse one or more dimensions
3. **Matrix multiplication** — combine dimensions
4. **Broadcasting** — operate on different shapes
5. **Shape manipulation** — rearrange without changing data

### Category 1: Element-wise Operations

Every element is processed **independently**. Input shape = Output shape.

```
Tensor A:  [1, 2, 3]
Tensor B:  [4, 5, 6]
─────────────────────
A + B:     [5, 7, 9]       ← each pair added independently
A * B:     [4, 10, 18]     ← each pair multiplied
A ** 2:    [1, 4, 9]       ← each element squared
```

Visual for 2D:

```
    [1, 2]       [10, 20]       [11, 22]
    [3, 4]   +   [30, 40]   =   [33, 44]

    (2, 2)       (2, 2)         (2, 2)
    ← same shape throughout →
```

Common element-wise operations:
```python
a + b       # addition
a - b       # subtraction
a * b       # multiplication (NOT matrix multiply!)
a / b       # division
a ** n      # power
torch.exp(a)    # e^x for each element
torch.log(a)    # ln(x) for each element
torch.relu(a)   # max(0, x) for each element
torch.sigmoid(a) # 1/(1+e^-x) for each element
```

**Key insight**: Element-wise ops NEVER change the shape.

### Category 2: Reduction Operations

**Collapse** one or more dimensions by aggregating values.

```
Input:  (3, 4)  — a 3×4 table
                                    
Sum along axis 0 (collapse rows):   output shape (4,)
Sum along axis 1 (collapse cols):   output shape (3,)
Sum all:                            output shape ()  — scalar
```

Visual:

```
        col0  col1  col2  col3
row 0: [  1,    2,    3,    4 ]     sum(axis=1) → [10]
row 1: [  5,    6,    7,    8 ]                    [26]
row 2: [  9,   10,   11,   12 ]                    [42]
        ─────────────────────
sum(axis=0): [15,  18,   21,  24]
```

```python
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]], dtype=torch.float)

x.sum()          # 78 — scalar, all elements
x.sum(dim=0)     # [15, 18, 21, 24] — collapse rows → shape (4,)
x.sum(dim=1)     # [10, 26, 42]     — collapse cols → shape (3,)
x.mean(dim=0)    # [5, 6, 7, 8]     — average down rows
x.max(dim=1)     # values=[4, 8, 12], indices=[3, 3, 3]
```

**The rule**: The dimension you specify in `dim=` **disappears** from the output shape.

```
Input shape:  (3, 4)
sum(dim=0):   (4,)     ← axis 0 (size 3) disappeared
sum(dim=1):   (3,)     ← axis 1 (size 4) disappeared
```

With `keepdim=True`, the dimension stays but becomes size 1:
```python
x.sum(dim=1, keepdim=True)  # shape (3, 1) instead of (3,)
```

### Category 3: Matrix Multiplication

The most important operation in deep learning. Combines two dimensions.

```
(M, K) @ (K, N) → (M, N)

The inner dimensions (K) must match and they "collapse."
```

Visual:

```
    A (2, 3)          B (3, 4)          C (2, 4)
┌───────────┐    ┌──────────────┐    ┌──────────────┐
│ a b c     │    │ w x y z      │    │ aw+be+ci ... │
│ d e f     │  @ │ . . . .      │  = │ dw+ee+fi ... │
└───────────┘    │ . . . .      │    └──────────────┘
                 └──────────────┘
```

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B              # shape (2, 4)
# or: C = torch.matmul(A, B)
```

**Batch matrix multiplication** — extra dimensions are "batch" dimensions:

```python
# (batch, M, K) @ (batch, K, N) → (batch, M, N)
A = torch.randn(8, 2, 3)
B = torch.randn(8, 3, 4)
C = A @ B              # shape (8, 2, 4) — 8 independent matmuls
```

**Why this matters**: Every neural network layer is basically:
```
output = input @ weights + bias
(64, 784) @ (784, 128) + (128,) → (64, 128)
```

### Category 4: Broadcasting

Operate on tensors of **different shapes** by virtually expanding the smaller one.

#### The 3 Rules of Broadcasting

```
1. Compare shapes from the RIGHT (last dimension first)
2. Dimensions must be EQUAL, or one of them must be 1
3. Missing dimensions are treated as 1 (prepended on the left)
```

#### Examples:

```
(3, 4) + (4,)     → (3, 4)    ✅  Rule 3: (4,) becomes (1, 4)
                                    Rule 2: 1 broadcasts to 3
                                    Result: (3, 4)

(3, 4) + (3, 1)   → (3, 4)    ✅  Rule 2: 1 broadcasts to 4

(3, 4) + (3,)     → ERROR     ❌  Right-aligned: 4 ≠ 3, neither is 1

(2, 1, 4) + (3, 4) → (2, 3, 4) ✅  (3, 4) becomes (1, 3, 4)
                                      then 1 broadcasts to 2
```

#### Visual:

```
Broadcasting (3, 1) + (1, 4):

    [1]         [10, 20, 30, 40]         [11, 21, 31, 41]
    [2]    +                          =   [12, 22, 32, 42]
    [3]                                   [13, 23, 33, 43]
   (3,1)           (1, 4)                    (3, 4)

The column [1,2,3] is "stretched" across 4 columns
The row [10,20,30,40] is "stretched" down 3 rows
```

**No data is actually copied** — PyTorch uses strides of 0 to "fake" the expansion.

### Category 5: Shape Manipulation

Rearrange elements without changing values.

```python
x = torch.arange(12)  # [0, 1, 2, ..., 11]

# Reshape — change shape
x.view(3, 4)       # (12,) → (3, 4)
x.reshape(2, 6)    # (12,) → (2, 6)

# Transpose — swap two axes
y = torch.randn(3, 4)
y.t()              # (3, 4) → (4, 3)
y.transpose(0, 1)  # same thing

# Permute — reorder all axes
z = torch.randn(2, 3, 4)
z.permute(2, 0, 1)  # (2, 3, 4) → (4, 2, 3)

# Squeeze — remove dimensions of size 1
a = torch.randn(1, 3, 1, 4)
a.squeeze()        # (1, 3, 1, 4) → (3, 4)

# Unsqueeze — add a dimension of size 1
b = torch.randn(3, 4)
b.unsqueeze(0)     # (3, 4) → (1, 3, 4)
b.unsqueeze(-1)    # (3, 4) → (3, 4, 1)

# Flatten — collapse dimensions
c = torch.randn(2, 3, 4)
c.flatten()            # (2, 3, 4) → (24,)
c.flatten(start_dim=1) # (2, 3, 4) → (2, 12)
```

### In-Place Operations (The Underscore Convention)

Operations ending with `_` modify the tensor **in place** (no new memory allocated):

```python
x = torch.tensor([1.0, 2.0, 3.0])

x.add_(10)      # x is now [11, 12, 13] — modified in place
x.zero_()       # x is now [0, 0, 0]
x.fill_(5)      # x is now [5, 5, 5]
x.mul_(2)       # x is now [10, 10, 10]
```

⚠️ **Warning**: In-place ops can break autograd. Avoid them during training unless you know what you're doing.

### The Complete Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│                    TENSOR OPERATIONS                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Element-wise:  same shape → same shape                     │
│                 +, -, *, /, **, relu, sigmoid               │
│                                                             │
│  Reduction:     collapse a dimension                        │
│                 sum, mean, max, min, argmax                  │
│                 dim= specifies WHICH axis disappears        │
│                                                             │
│  MatMul:        combine two dimensions                      │
│                 (M,K) @ (K,N) → (M,N)                      │
│                 inner dims must match                        │
│                                                             │
│  Broadcasting:  different shapes, no copy                   │
│                 align from right, 1s expand                  │
│                                                             │
│  Reshape:       rearrange, no value change                  │
│                 view, reshape, permute, squeeze              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### ⚠️ Common Mistake

**Confusing `*` (element-wise) with `@` (matrix multiply)**

```python
A = torch.randn(3, 4)
B = torch.randn(3, 4)

A * B   # element-wise multiply → shape (3, 4)
A @ B   # ERROR! (3,4) @ (3,4) — inner dims don't match

A @ B.T  # matrix multiply → (3,4) @ (4,3) = (3,3) ✅
```

---


## Section 6: Tensors in Deep Learning Context

### In Plain English

In deep learning, there are only **4 types of tensors** you'll ever deal with:

1. **Input tensors** — your data
2. **Weight tensors** — the model's learnable parameters
3. **Output tensors** — predictions
4. **Gradient tensors** — how to update weights

Everything else is just these 4 flowing through operations.

### The Full Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   INPUT          WEIGHTS         OUTPUT         LOSS        GRADS   │
│   (your data)    (learned)       (prediction)   (scalar)    (∂L/∂W) │
│                                                                     │
│   ┌─────┐       ┌─────┐        ┌─────┐       ┌─────┐    ┌─────┐  │
│   │64,784│ ──→  │784, │  ──→   │64,  │  ──→  │  1  │    │784, │  │
│   │     │   @   │ 128 │   =    │ 128 │  CE   │     │    │ 128 │  │
│   └─────┘       └─────┘        └─────┘       └─────┘    └─────┘  │
│                                                    │          ↑     │
│                                                    └──────────┘     │
│                                                    .backward()      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. Input Tensors (Your Data)

These are the tensors you create from your dataset.

```python
# Image batch
images = torch.randn(64, 3, 224, 224)  # 64 RGB images, 224×224
# Properties:
#   requires_grad = False (we don't learn the input)
#   device = 'cuda' (moved to GPU for speed)

# Text batch
tokens = torch.randint(0, 50000, (32, 128))  # 32 sentences, 128 tokens each
# Properties:
#   dtype = torch.long (integer indices into vocabulary)
```

### 2. Weight Tensors (Learnable Parameters)

These are inside your model. PyTorch creates them when you define layers.

```python
layer = nn.Linear(784, 128)

# This creates TWO weight tensors:
print(layer.weight.shape)  # (128, 784) — the W matrix
print(layer.bias.shape)    # (128,)     — the b vector

# Properties:
#   requires_grad = True (we LEARN these!)
#   initialized randomly
#   updated by optimizer after each backward pass
```

What a neural network "learns" = finding the right values for these weight tensors.

### 3. Output Tensors (Predictions)

Results of passing input through the model.

```python
model = nn.Linear(784, 10)
input_batch = torch.randn(64, 784)

output = model(input_batch)  # shape (64, 10)
# 64 samples, each with 10 scores (one per class)

# Properties:
#   requires_grad = True (because it depends on weights that have requires_grad=True)
#   has a grad_fn (records how it was computed)
```

### 4. Gradient Tensors (How to Update)

After calling `.backward()`, every weight tensor gets a `.grad` attribute — same shape as the weight, telling it which direction to move.

```python
loss.backward()

# Now every weight has a gradient:
print(layer.weight.grad.shape)  # (128, 784) — same shape as weight!
print(layer.bias.grad.shape)    # (128,)     — same shape as bias!

# The gradient says: "if you increase this weight by 1,
#                     the loss changes by this much"
```

### The Training Cycle — All Tensors Together

```python
# Step 1: Input tensor flows forward
input = torch.randn(64, 784)           # INPUT TENSOR
output = model(input)                   # OUTPUT TENSOR (uses WEIGHT TENSORS internally)

# Step 2: Compare output to target → scalar loss
target = torch.randint(0, 10, (64,))
loss = criterion(output, target)        # LOSS (scalar tensor)

# Step 3: Compute gradients
loss.backward()                         # GRADIENT TENSORS created (stored in .grad)

# Step 4: Update weights using gradients
optimizer.step()                        # WEIGHT TENSORS modified
optimizer.zero_grad()                   # GRADIENT TENSORS zeroed
```

### Tensor Properties That Matter

```python
x = torch.randn(3, 4, requires_grad=True)

x.data           # the raw numbers (without grad tracking)
x.grad           # the gradient (None until .backward() is called)
x.grad_fn        # the function that created this tensor (None for leaf tensors)
x.requires_grad  # True = track operations for gradient computation
x.is_leaf        # True = created by user (not result of an operation)
```

### Which Tensors Have requires_grad=True?

```
Model weights:     requires_grad = True   (we learn these)
Model biases:      requires_grad = True   (we learn these)
Input data:        requires_grad = False  (we don't learn inputs)
Labels/targets:    requires_grad = False  (we don't learn labels)
Intermediate:      requires_grad = True   (inherited from weights)
Loss:              requires_grad = True   (inherited from weights)
```

### The Computation Graph — Tensors Connected

Every operation creates a node in the graph. PyTorch walks this graph backward to compute gradients.

```
x (leaf, input)──────────┐
                          ▼
                    z = x @ W + b ──→ relu(z) ──→ output ──→ loss
                          ▲                                     │
W (leaf, weight)─────────┘                                     │
                                                               │
                    loss.backward() walks back: ◄──────────────┘
                    computes ∂loss/∂W and stores in W.grad
```

### Tensor Lifecycle in One Training Step

```
1. BIRTH:     Input tensor created from DataLoader
2. FORWARD:   Flows through layers, multiplied by weight tensors
3. LOSS:      Output compared to target → loss tensor (scalar)
4. BACKWARD:  Gradient tensors computed for all weights
5. UPDATE:    Weight tensors modified by optimizer
6. CLEANUP:   Gradient tensors zeroed, computation graph destroyed
7. REPEAT:    Next batch...
```

---

### ⚠️ Common Mistake

**Trying to call .numpy() on a tensor that requires grad or is on GPU**

```python
x = torch.randn(3, requires_grad=True, device='cuda')

# ❌ WRONG
# x.numpy()  — ERROR! Can't convert CUDA tensor to numpy
# x.numpy()  — ERROR! Can't convert tensor with grad to numpy

# ✅ CORRECT
x.detach().cpu().numpy()
# detach() → removes from computation graph
# cpu()    → moves to CPU
# numpy()  → converts to numpy array
```

---


## Section 7: Tensor Gotchas & Intuition Builders

### In Plain English

This section is about the **mistakes you WILL make** and how to fix them fast. Plus, mental tricks to "think in tensors."

---

### Gotcha 1: Shape Mismatches (The #1 Error)

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x784 and 256x128)
```

**How to debug:**

```python
# Print shapes at every step
print(f"input: {x.shape}")
x = layer1(x)
print(f"after layer1: {x.shape}")
x = layer2(x)
print(f"after layer2: {x.shape}")
```

**Common causes:**

```python
# 1. Forgot to flatten before FC layer
x = torch.randn(64, 3, 32, 32)  # images
# x = linear(x)  ← ERROR! Linear expects (batch, features), not (batch, C, H, W)
x = x.flatten(start_dim=1)       # (64, 3072)
x = linear(x)                    # now works

# 2. Wrong dimension in Linear layer
nn.Linear(784, 128)  # expects input with LAST dim = 784
# If your input is (64, 256), this will fail
# Fix: nn.Linear(256, 128)

# 3. Mismatched batch sizes
a = torch.randn(32, 10)
b = torch.randn(64, 10)
# a + b  ← ERROR! batch sizes 32 ≠ 64
```

**The golden rule**: Before writing any model, trace the shapes on paper:

```
Input:          (64, 1, 28, 28)
After Conv1:    (64, 32, 28, 28)    ← padding=1 keeps size
After Pool1:    (64, 32, 14, 14)    ← halved
After Conv2:    (64, 64, 14, 14)
After Pool2:    (64, 64, 7, 7)
After Flatten:  (64, 3136)          ← 64 * 7 * 7 = 3136
After FC1:      (64, 128)
After FC2:      (64, 10)            ← 10 classes
```

---

### Gotcha 2: Device Mismatches

```
RuntimeError: Expected all tensors to be on the same device
```

**Rule**: ALL tensors in an operation must be on the SAME device.

```python
model = model.to('cuda')
input = input.to('cuda')
target = target.to('cuda')
# Now everything is on GPU — operations work

# ❌ Common mistake: creating a new tensor without specifying device
mask = torch.zeros(64, 10)  # this is on CPU!
output = model(input)        # this is on GPU!
# result = output * mask     # ERROR! different devices

# ✅ Fix:
mask = torch.zeros(64, 10, device=input.device)
```

---

### Gotcha 3: dtype Mismatches

```
RuntimeError: expected scalar type Float but found Long
```

**Common situations:**

```python
# Labels must be Long (int64) for CrossEntropyLoss
labels = torch.tensor([0, 1, 2])          # default: int64 ✅

# Features must be Float for neural networks
features = torch.tensor([1, 2, 3])        # default: int64 ❌
features = torch.tensor([1, 2, 3], dtype=torch.float32)  # ✅
# or: features = features.float()

# Mixing int and float
a = torch.tensor([1, 2, 3])       # int64
b = torch.tensor([1.0, 2.0, 3.0]) # float32
# c = nn.Linear(3, 1)(a)  ← ERROR! Linear needs float
# c = nn.Linear(3, 1)(a.float())  ← works
```

**Quick reference:**
```
Model inputs:  float32 (always)
Labels:        int64 / long (for classification)
               float32 (for regression)
Indices:       int64 / long (for embedding lookups)
```

---

### Gotcha 4: Detach, Clone, Contiguous

Three operations that confuse everyone:

```python
x = torch.randn(3, 4, requires_grad=True)

# .detach() — removes from computation graph (no more gradient tracking)
y = x.detach()
# Use when: you want to use a tensor's VALUES without affecting gradients
# Example: logging loss value, using output as input to another model

# .clone() — creates a TRUE copy (new memory)
z = x.clone()
# Use when: you need an independent copy that won't affect the original
# Note: clone() preserves requires_grad

# .contiguous() — rearranges memory to be contiguous
w = x.t().contiguous()
# Use when: you get "not contiguous" errors after transpose/permute
# Note: if already contiguous, this is a no-op (free)
```

**When you need combinations:**

```python
# "Give me the values, independent copy, no grad"
safe_copy = x.detach().clone()

# "Move to numpy safely"
numpy_arr = x.detach().cpu().numpy()
```

---

### Gotcha 5: In-place Operations Breaking Autograd

```python
x = torch.randn(3, requires_grad=True)
y = x * 2

# ❌ This breaks the computation graph:
# y.add_(1)  # in-place modification of y
# loss = y.sum()
# loss.backward()  # ERROR! graph was modified in-place

# ✅ Use out-of-place instead:
y = y + 1  # creates new tensor, graph intact
loss = y.sum()
loss.backward()  # works
```

---

### Intuition Builder 1: Think in Shapes, Not Values

When designing a model, **ignore the actual numbers**. Only think about shapes.

```
"I have 64 images of size 224×224 with 3 channels"
→ Shape: (64, 3, 224, 224)

"I need to classify into 10 categories"
→ Output shape: (64, 10)

"What operations transform (64, 3, 224, 224) → (64, 10)?"
→ Conv layers reduce spatial dims
→ Flatten converts to (64, N)
→ Linear converts (64, N) → (64, 10)
```

---

### Intuition Builder 2: Dimension = "For Each"

Read each dimension as "for each ___":

```
Shape: (64, 50, 768)

Read as: "For each of 64 sentences,
          for each of 50 tokens,
          there's a 768-dimensional vector"
```

```
Shape: (32, 3, 224, 224)

Read as: "For each of 32 images,
          for each of 3 color channels,
          there's a 224×224 grid of values"
```

---

### Intuition Builder 3: dim= Means "Collapse Along"

When you see `dim=0` in a reduction, think "collapse that axis":

```
x.shape = (3, 4)

x.sum(dim=0)  → "collapse the rows (axis 0), keep columns"  → shape (4,)
x.sum(dim=1)  → "collapse the columns (axis 1), keep rows"  → shape (3,)
```

Another way to think: **dim= is the dimension that DISAPPEARS**.

---

### Intuition Builder 4: Broadcasting = "Stretch the 1s"

```
(3, 1) + (1, 4) → (3, 4)

"Stretch the column (3,1) across 4 columns"
"Stretch the row (1,4) down 3 rows"
"Now both are (3,4) — add element-wise"
```

---

### Intuition Builder 5: Matrix Multiply = "Match Inner, Keep Outer"

```
(M, K) @ (K, N) → (M, N)
       ↑     ↑
       └─────┘
    These must match (and disappear)

"I have M things, each described by K features"
"I have a transformation from K features to N features"
"Result: M things, each now described by N features"
```

---

### Quick Debugging Checklist

When something goes wrong with tensors, check these in order:

```
□ Shape:   print(tensor.shape) — is it what you expect?
□ Device:  print(tensor.device) — are all tensors on same device?
□ Dtype:   print(tensor.dtype) — float32 for model input?
□ Values:  print(tensor.min(), tensor.max()) — any NaN/Inf?
□ Grad:    print(tensor.requires_grad) — should this track gradients?
□ Memory:  print(tensor.is_contiguous()) — needed for .view()?
```

---

### Final Summary: The Tensor Mental Model

```
┌─────────────────────────────────────────────────────────────────┐
│                         A TENSOR IS:                             │
│                                                                 │
│   • A multi-dimensional container of numbers                    │
│   • Defined by: shape + dtype + device                          │
│   • Lives in flat memory, accessed via strides                  │
│   • Can track operations for automatic gradient computation     │
│   • Can live on GPU for parallel computation                    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     YOU OPERATE ON TENSORS BY:                   │
│                                                                 │
│   • Element-wise ops (same shape → same shape)                  │
│   • Reductions (collapse dimensions)                            │
│   • Matrix multiply (combine dimensions)                        │
│   • Broadcasting (expand 1s to match shapes)                    │
│   • Reshaping (rearrange without changing data)                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                  IN DEEP LEARNING, TENSORS ARE:                  │
│                                                                 │
│   • Inputs (your data, no grad)                                 │
│   • Weights (learned parameters, has grad)                      │
│   • Outputs (predictions, computed from inputs × weights)       │
│   • Gradients (direction to update weights)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**You now have a complete mental model of tensors. Go run the Phase 1 notebook and everything will click.** 🚀
