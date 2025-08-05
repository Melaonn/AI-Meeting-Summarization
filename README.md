# LLM 4-bit Quantization Project

A lightweight implementation exploring 4-bit quantization techniques for Large Language Models (LLMs) using BitsAndBytes library. This project serves as a foundation for learning advanced LLM optimization techniques.

## 🎯 Project Overview

This project demonstrates efficient model loading and inference using 4-bit quantization with double quantization enabled. The implementation focuses on memory optimization while maintaining model performance.

## ✨ Features

- **4-bit Quantization**: Reduces memory usage by ~75% compared to FP16
- **Double Quantization**: Further compression using nested quantization
- **NF4 Quantization Type**: Optimal 4-bit format for neural network weights
- **BFloat16 Compute**: Maintains numerical stability during computations

## 🛠️ Technical Configuration

```python
quantization_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_quant_type": "nf4"
}
```

### Configuration Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| `load_in_4bit` | `True` | Enables 4-bit quantization |
| `bnb_4bit_use_double_quant` | `True` | Applies quantization to quantization constants |
| `bnb_4bit_compute_dtype` | `torch.bfloat16` | Data type for computations |
| `bnb_4bit_quant_type` | `"nf4"` | Normal Float 4-bit quantization |


### Benefits

- ✅ **Memory Efficient**: ~75% reduction in GPU memory usage
- ✅ **Faster Loading**: Reduced model loading time
- ✅ **Cost Effective**: Run larger models on smaller hardware
- ⚠️ **Slight Quality Trade-off**: Minimal impact on model performance


## 🔬 Advanced Techniques (Future Learning)

This project is designed to evolve. Planned advanced techniques include:

- [ ] **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
- [ ] **QLoRA**: Quantized LoRA for even more efficiency
- [ ] **GPTQ**: Advanced quantization techniques
- [ ] **Model Pruning**: Removing redundant parameters
- [ ] **Knowledge Distillation**: Teacher-student model training
- [ ] **Dynamic Quantization**: Runtime quantization optimization

# Ensure CUDA toolkit is installed
pip install bitsandbytes --
