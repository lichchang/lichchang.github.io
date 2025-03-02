---
title: 'Fake Model Quantization'
date: 2025-02-25
permalink: /posts/2025/02/Fake-Model-Quantization/
tags:
  - Model Quantization
  - Post-training quantization
---



# Fake Model Quantization Doesn't Make Any Difference in Accelerating Model Inference Time

## Introduction

Quantization (specifically referring to post-training quantization in this article) is a widely used technique to reduce the memory footprint and computational cost of AI models. However, a common misconception is that **fake quantization** can directly lead to an acceleration in inference time. In this article, we explore why fake quantization does not yield inference speedup and what actually contributes to runtime improvements.

## Fake Quantization & Real quantization

***Fake quantization*** is a technique used to simulate the effects of low-bit precision computations while still using floating-point arithmetic. This helps the model learn to be robust to quantization errors before actual deployment in integer precision. It inserts quantization and dequantization operations into the model layers while still using floating-point precision. This allows the model to adapt to lower-bit representations before actual deployment. In other words, quantization implementations that don’t use additional CUDA tools to support low-bit matrix manipulation are all considered 'fake' ones.



***Real quantization***, unlike fake quantization, converts trained weights and activations to lower precision at inference time. This is done by replacing each layer in the model with layers that incorporate CUDA-supported matrix manipulation (GEMM). As you can see below, every layer has been replaced with a GEMM version.


A self-attention block in the OPT model. It would be exactly the same as this one when applied after fake quantization.
``` 
OPTDecoderLayer(
  (self_attn): OPTAttention(
    (k_proj): Linear(in_features=2560, out_features=2560, bias=True)
    (v_proj): Linear(in_features=2560, out_features=2560, bias=True)
    (q_proj): Linear(in_features=2560, out_features=2560, bias=True)
    (out_proj): Linear(in_features=2560, out_features=2560, bias=True)
  )
  (activation_fn): ReLU()
  (self_attn_layer_norm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  (fc1): Linear(in_features=2560, out_features=10240, bias=True)
  (fc2): Linear(in_features=10240, out_features=2560, bias=True)
  (final_layer_norm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
)
```

A real quantization version (applied SmoothQuant W8A8 [^1]):
[^1]:https://github.com/mit-han-lab/smoothquant
``` 
Int8OPTDecoderLayer(
  (self_attn): Int8OPTAttention(
    (qk_bmm): BMM_S8T_S8N_F32T()
    (pv_bmm): BMM_S8T_S8N_S8T()
    (k_proj): W8A8B8O8Linear()
    (v_proj): W8A8B8O8Linear()
    (q_proj): W8A8B8O8Linear()
    (out_proj): W8A8BFP32OFP32Linear()
  )
  (self_attn_layer_norm): LayerNormQ()
  (fc1): W8A8B8O8LinearReLU()
  (fc2): W8A8BFP32OFP32Linear()
  (final_layer_norm): LayerNormQ()
)
```



**Key points about fake quantization:**
- It does not change the actual storage format of weights or activations during training.
- Computation is still performed in floating-point precision (e.g., FP32 or BF16).
- The model structure remains unchanged, meaning no real inference-time optimization occurs.


**Key points about real quantization:**
- It changes the storage format and activation values to lower-bit representations (e.g., Int8, Int4).
- All computations are performed using GEMM tools, which support low-precision operations (e.g. BMM_S8T_S8N_F32T).
- The entire model structure is rewritten into a tailored low-bit version.

