# Part 2: NumPy Arrays — Fast Math Without Loops

---

## The Problem NumPy Solves

Python lists are slow for math because:
1. Each element is a separate Python object
2. You need explicit loops for every operation
3. No hardware-level optimization

NumPy fixes ALL of this.

---

## What is a NumPy Array?

A NumPy array (ndarray) is a **fixed-type, fixed-size block of numbers** stored in contiguous memory.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
```

Key differences from lists:

```
Python List:     [  obj1  |  obj2  |  obj3  |  obj4  ]
                   ↓         ↓         ↓         ↓
                 int(1)    int(2)    int(3)    int(4)
                 28 bytes  28 bytes  28 bytes  28 bytes
                 (scattered in memory)

NumPy Array:     [ 1 | 2 | 3 | 4 ]
                  4B  4B  4B  4B
                 (one continuous block, no overhead)
```

---

## Where Do We Use NumPy?

### 1. Data Science & Analytics

```python
# Sales data for 365 days
sales = np.array([...])  # shape (365,)

# Average, max, trend
print(sales.mean())       # no loop needed
print(sales.max())
print(sales[sales > 1000])  # filter: days with sales > 1000
```

### 2. Image Processing

```python
# An image is just a grid of numbers
image = np.zeros((480, 640, 3), dtype=np.uint8)  # black 640x480 RGB image

# Make it red
image[:, :, 0] = 255  # set red channel to max

# Crop
cropped = image[100:300, 200:500]  # slice rows and columns

# Flip
flipped = image[:, ::-1]  # reverse columns = horizontal flip
```

### 3. Scientific Computing

```python
# Solve physics: projectile motion
t = np.linspace(0, 10, 1000)  # 1000 time points
g = 9.81
v0 = 50  # initial velocity

y = v0 * t - 0.5 * g * t**2  # height at each time — NO LOOP!
```

### 4. Statistics & Probability

```python
# Generate 1 million random samples
samples = np.random.normal(mean=0, std=1, size=1_000_000)

print(f"Mean: {samples.mean():.4f}")      # ≈ 0
print(f"Std:  {samples.std():.4f}")       # ≈ 1
print(f"P(x > 2): {(samples > 2).mean():.4f}")  # ≈ 0.0228
```

### 5. Linear Algebra

```python
# Solve system of equations: Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 10])
x = np.linalg.solve(A, b)  # x = [1, 3]

# Eigenvalues
eigenvalues = np.linalg.eigvals(A)
```

### 6. Signal Processing

```python
# Audio waveform (1 second at 44100 Hz)
t = np.linspace(0, 1, 44100)
frequency = 440  # Hz (A note)
waveform = np.sin(2 * np.pi * frequency * t)

# Mix two frequencies
chord = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 554 * t)
```

---

## How NumPy Works — The Magic

### Vectorization (No Loops!)

```python
# Python list way (SLOW)
a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]
result = [a[i] + b[i] for i in range(5)]

# NumPy way (FAST)
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])
result = a + b  # [7, 9, 11, 13, 15] — done in C, not Python!
```

Why is this fast?
- The loop happens in **compiled C code**, not interpreted Python
- Data is **contiguous in memory** (CPU cache friendly)
- Operations use **SIMD instructions** (process 4-8 numbers at once)

### Element-wise Operations (Just Work™)

```python
a = np.array([1, 2, 3, 4])

a + 10       # [11, 12, 13, 14]
a * 2        # [2, 4, 6, 8]
a ** 2       # [1, 4, 9, 16]
np.sqrt(a)   # [1.0, 1.41, 1.73, 2.0]
np.exp(a)    # [2.71, 7.38, 20.08, 54.59]
a > 2        # [False, False, True, True]
```

### Shape Awareness

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print(matrix.shape)    # (2, 3) — 2 rows, 3 columns
print(matrix.ndim)     # 2 — two dimensions
print(matrix.size)     # 6 — total elements
print(matrix.dtype)    # int64 — all elements same type
```

### Broadcasting

```python
# Add a row vector to every row of a matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])  # (3, 3)

row = np.array([10, 20, 30])    # (3,)

result = matrix + row  # row broadcasts to all rows!
# [[11, 22, 33],
#  [14, 25, 36],
#  [17, 28, 39]]
```

### Slicing & Indexing (Powerful)

```python
a = np.arange(20).reshape(4, 5)
# [[ 0,  1,  2,  3,  4],
#  [ 5,  6,  7,  8,  9],
#  [10, 11, 12, 13, 14],
#  [15, 16, 17, 18, 19]]

a[0]          # first row: [0, 1, 2, 3, 4]
a[:, 0]       # first column: [0, 5, 10, 15]
a[1:3, 2:4]   # submatrix: [[7, 8], [12, 13]]
a[a > 10]     # all elements > 10: [11, 12, 13, 14, 15, 16, 17, 18, 19]
```

---

## Real-World Example: Analyzing Student Grades

```python
# 100 students, 5 subjects
grades = np.random.randint(50, 100, size=(100, 5))

# Average grade per student
student_avg = grades.mean(axis=1)  # shape (100,)

# Average grade per subject
subject_avg = grades.mean(axis=0)  # shape (5,)

# Top students (average > 85)
top_students = np.where(student_avg > 85)[0]

# Normalize grades (z-score)
normalized = (grades - grades.mean(axis=0)) / grades.std(axis=0)

# All of this: NO LOOPS. Runs in milliseconds on millions of rows.
```

---

## Why NumPy is NOT Enough for Deep Learning

NumPy is incredible for CPU computation. But it has limits:

| Limitation | Why It Matters for DL |
|-----------|----------------------|
| **CPU only** | Neural networks need GPU (1000x faster for matrix math) |
| **No auto-differentiation** | Can't compute gradients automatically |
| **No computation graph** | Can't do backpropagation |
| **No GPU memory management** | Can't move data to/from GPU |

### The speed gap:

```
Matrix multiply (4096 × 4096):
  NumPy (CPU):        ~2000 ms
  PyTorch (CPU):      ~2000 ms  (same on CPU)
  PyTorch (GPU):      ~5 ms     (400x faster!)
```

For training a neural network with millions of parameters on millions of data points, **CPU is simply too slow**.

---

## Summary

```
NumPy Array:
  ✅ Fast (C-compiled, vectorized, SIMD)
  ✅ Element-wise math without loops
  ✅ Shape-aware (knows it's 3×4, not just 12 numbers)
  ✅ Fixed dtype (memory efficient)
  ✅ Broadcasting (operate on different shapes)
  ✅ Rich ecosystem (scipy, pandas, matplotlib all use it)
  
  ❌ CPU only (no GPU acceleration)
  ❌ No automatic differentiation
  ❌ No computation graph for backpropagation
  ❌ Not designed for deep learning workflows
```

**Next → Part 3: Matrices — The Math Behind Everything**
