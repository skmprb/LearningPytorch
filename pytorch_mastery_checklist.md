# PyTorch Mastery Checklist

## Phase 1: Foundations
- [ ] Install PyTorch and verify GPU support (`torch.cuda.is_available()`)
- [ ] Understand tensors: creation, dtypes, shapes
- [ ] Tensor operations: indexing, slicing, reshaping (`view`, `reshape`, `permute`)
- [ ] Tensor math: element-wise ops, matrix multiplication (`@`, `matmul`, `bmm`)
- [ ] Broadcasting rules
- [ ] NumPy ↔ Tensor conversion (`from_numpy`, `.numpy()`)
- [ ] Move tensors between CPU/GPU (`.to()`, `.cuda()`, `.cpu()`)

## Phase 2: Autograd & Differentiation
- [ ] `requires_grad`, `.backward()`, `.grad`
- [ ] Computation graph (dynamic vs static)
- [ ] `torch.no_grad()` and `detach()`
- [ ] Gradient accumulation and zeroing
- [ ] Custom autograd functions (`torch.autograd.Function`)

## Phase 3: Neural Network Building Blocks
- [ ] `nn.Module` — build custom models
- [ ] Common layers: `Linear`, `Conv2d`, `BatchNorm`, `Dropout`
- [ ] Activation functions: `ReLU`, `GELU`, `Sigmoid`, `Softmax`
- [ ] `nn.Sequential` vs custom `forward()`
- [ ] Parameter inspection (`.parameters()`, `.named_parameters()`)
- [ ] Weight initialization (`nn.init`)

## Phase 4: Data Pipeline
- [ ] `torch.utils.data.Dataset` — custom datasets
- [ ] `torch.utils.data.DataLoader` — batching, shuffling, num_workers
- [ ] `torchvision.transforms` — image augmentations
- [ ] `torchvision.datasets` — built-in datasets (MNIST, CIFAR10)
- [ ] `random_split` for train/val splits
- [ ] Handling imbalanced data (`WeightedRandomSampler`)

## Phase 5: Training Loop Mastery
- [ ] Write a full training loop from scratch
- [ ] Loss functions: `CrossEntropyLoss`, `MSELoss`, `BCEWithLogitsLoss`
- [ ] Optimizers: `SGD`, `Adam`, `AdamW`
- [ ] Learning rate schedulers (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`)
- [ ] Validation loop and metric tracking
- [ ] Early stopping implementation
- [ ] Gradient clipping (`torch.nn.utils.clip_grad_norm_`)

## Phase 6: Model Save/Load & Inference
- [ ] Save/load model weights (`state_dict`)
- [ ] Save/load full checkpoints (model + optimizer + epoch)
- [ ] `model.eval()` vs `model.train()` modes
- [ ] `torch.no_grad()` for inference
- [ ] Export to ONNX (`torch.onnx.export`)
- [ ] TorchScript (`torch.jit.trace`, `torch.jit.script`)

## Phase 7: CNNs (Computer Vision)
- [ ] Build a CNN from scratch (Conv → Pool → FC)
- [ ] Train on CIFAR-10 or MNIST
- [ ] Transfer learning with pretrained models (`torchvision.models`)
- [ ] Fine-tuning vs feature extraction
- [ ] Data augmentation strategies
- [ ] Visualize filters and feature maps

## Phase 8: RNNs & Sequence Models
- [ ] `nn.RNN`, `nn.LSTM`, `nn.GRU`
- [ ] Packed sequences (`pack_padded_sequence`)
- [ ] Bidirectional RNNs
- [ ] Sequence-to-sequence architecture
- [ ] Text classification project

## Phase 9: Transformers & Attention
- [ ] Scaled dot-product attention from scratch
- [ ] Multi-head attention (`nn.MultiheadAttention`)
- [ ] Positional encoding
- [ ] `nn.Transformer` module
- [ ] Build a mini GPT or BERT-style model
- [ ] Integration with Hugging Face `transformers`

## Phase 10: Advanced Training Techniques
- [ ] Mixed precision training (`torch.cuda.amp`, `autocast`, `GradScaler`)
- [ ] Gradient accumulation for large effective batch sizes
- [ ] Distributed training (`DistributedDataParallel`)
- [ ] `DataParallel` vs `DistributedDataParallel`
- [ ] Multi-GPU strategies
- [ ] Profiling with `torch.profiler`

## Phase 11: Generative Models
- [ ] Autoencoders (AE) and Variational Autoencoders (VAE)
- [ ] GANs — Generator + Discriminator training loop
- [ ] DCGAN implementation
- [ ] Diffusion models basics

## Phase 12: Debugging & Best Practices
- [ ] Debugging with `torchinfo` (model summary)
- [ ] Detect anomalies (`torch.autograd.set_detect_anomaly`)
- [ ] Reproducibility (`torch.manual_seed`, deterministic mode)
- [ ] Memory management (`torch.cuda.empty_cache()`, memory profiling)
- [ ] Common pitfalls (forgetting `zero_grad`, wrong tensor shapes, data leakage)
- [ ] Unit testing models (shape checks, overfitting a single batch)

## Phase 13: Deployment
- [ ] TorchServe for model serving
- [ ] Export to ONNX Runtime
- [ ] Mobile deployment with PyTorch Mobile
- [ ] AWS SageMaker integration for training & hosting
- [ ] Quantization (`torch.quantization`) for model compression

## Projects (Apply What You Learned)
- [ ] **P1:** MNIST digit classifier (Phase 1–5)
- [ ] **P2:** CIFAR-10 image classifier with transfer learning (Phase 7)
- [ ] **P3:** Sentiment analysis with LSTM (Phase 8)
- [ ] **P4:** Text generation with Transformer (Phase 9)
- [ ] **P5:** GAN for image generation (Phase 11)
- [ ] **P6:** Deploy a model to an API endpoint (Phase 13)
