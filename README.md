![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![TRL](https://img.shields.io/badge/TRL-Alignment-green)
![Status](https://img.shields.io/badge/Status-Active-success)

# LLM Alignment System

> Production-Style LLM Alignment Platform implementing Synthetic Data Generation, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), Evaluation, Research Analysis, Monitoring, and Deployment Workflows.

---

## Overview

LLM Alignment System is an end-to-end framework for studying and implementing modern language model alignment techniques.

The project follows a complete alignment pipeline:

```text
Synthetic Dataset Generation
          ↓
     Base Model
          ↓
 Baseline Evaluation
          ↓
Supervised Fine-Tuning (SFT)
          ↓
Direct Preference Optimization (DPO)
          ↓
Benchmark Comparison
          ↓
Research Analysis
          ↓
Production Deployment
```

The system combines LLM engineering, model alignment, evaluation science, research tooling, and MLOps practices into a single production-oriented project.

---

## Quick Start

### Clone Repository

```bash
git clone https://github.com/TechySan031/llm-alignment-system
cd llm-alignment-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Dataset

```bash
python scripts/generate_dataset.py --no-public
```

### Train SFT

```bash
python scripts/train_sft.py
```

### Train DPO

```bash
python scripts/train_dpo.py
```

### Evaluate Models

```bash
python scripts/evaluate.py --model-stage sft
python scripts/evaluate.py --model-stage dpo
```

## Key Features

### Alignment Pipeline

* Synthetic Dataset Generation
* Supervised Fine-Tuning (SFT)
* Direct Preference Optimization (DPO)
* Preference Dataset Construction
* LoRA / PEFT Adaptation

### Evaluation Framework

* Benchmarking Pipeline
* JSON Schema Validation
* Hallucination Detection
* Perplexity Analysis
* Cross-Model Comparison

### Research Tooling

* Attention Analysis
* Gradient Flow Analysis
* Layer Drift Measurement
* Representation Similarity
* Catastrophic Forgetting Studies

### Production Infrastructure

* FastAPI Inference Service
* GPU Monitoring
* Throughput & Latency Tracking
* Docker Deployment
* Kubernetes Infrastructure
* Model Registry

---

## System Architecture

![Architecture](docs/architecture.png)


```text
┌─────────────────────────────────────────────┐
│                DATA LAYER                   │
├─────────────────────────────────────────────┤
│ Synthetic Data Generation                   │
│ Schema Validation & Quality Control         │
│ Dataset Processing & Preparation            │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│               MODEL LAYER                   │
├─────────────────────────────────────────────┤
│ Qwen Foundation Models                      │
│ LoRA / PEFT Configuration                   │
│ Quantization & Memory Optimization          │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│             ALIGNMENT LAYER                 │
├─────────────────────────────────────────────┤
│ Baseline Evaluation                         │
│ Supervised Fine-Tuning (SFT)                │
│ Direct Preference Optimization (DPO)        │
│ Checkpointing & Experiment Tracking         │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│            EVALUATION LAYER                 │
├─────────────────────────────────────────────┤
│ Benchmarking                                │
│ JSON Schema Compliance                      │
│ Hallucination Analysis                      │
│ Perplexity & Quality Metrics                │
│ Base → SFT → DPO Comparison                 │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│             RESEARCH LAYER                  │
├─────────────────────────────────────────────┤
│ Attention Analysis                          │
│ Gradient Flow Analysis                      │
│ Layer Drift Measurement                     │
│ Representation Similarity                   │
│ Catastrophic Forgetting Studies             │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│             SERVING LAYER                   │
├─────────────────────────────────────────────┤
│ Inference Engine                            │
│ FastAPI Service                             │
│ Request Batching                            │
│ Streaming Responses                         │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           OBSERVABILITY LAYER               │
├─────────────────────────────────────────────┤
│ GPU Monitoring                              │
│ Latency Tracking                            │
│ Throughput Analytics                        │
│ Resource Utilization                        │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│            DEPLOYMENT LAYER                 │
├─────────────────────────────────────────────┤
│ Docker Containers                           │
│ Model Registry                              │
│ Health Monitoring                           │
│ Kubernetes & Autoscaling                    │
└─────────────────────────────────────────────┘
```

---

## Repository Structure

```text
configs/          Training and evaluation configs
data/             Generated datasets and schemas
deployment/       Deployment assets
docker/           Containerization files
docs/             Architecture diagrams and documentation
notebooks/        Research and analysis notebooks
scripts/          Training, evaluation, and utility scripts
src/              Core implementation
tests/            Automated test suite
outputs/          Evaluation and benchmark artifacts
```

## Technology Stack

### Core AI Stack

* PyTorch
* Transformers
* PEFT
* TRL
* Accelerate

### Data & Evaluation

* Hugging Face Datasets
* NumPy
* Pandas
* Scikit-Learn
* JSON Schema

### Monitoring & Deployment

* FastAPI
* Docker
* Kubernetes
* Prometheus
* Grafana

### Experiment Tracking

* Weights & Biases
* TensorBoard

---

## Alignment Workflow

```text
Generate 5,500 SFT Examples
            │
            ▼
Generate 3,151 DPO Pairs
            │
            ▼
 Qwen2.5-0.5B-Instruct
            │
            ▼
      SFT Training
            │
            ▼
     SFT Benchmark
      Alignment: 57.3%
 Hallucination: 2.0%
            │
            ▼
      DPO Training
            │
            ▼
     DPO Benchmark
      Alignment: 52.0%
 Hallucination: 15.0%
            │
            ▼
   Benchmark Comparison
            │
            ▼
 Alignment Research &
 Preference Optimization Analysis
            │
            ▼
   Inference Deployment
            │
            ▼
  Alignment Playground
```
---

## Benchmark Results
> **Note:** Results were obtained using Qwen2.5-0.5B-Instruct with LoRA-based fine-tuning and evaluation on a 200-example benchmark set.

Evaluation performed on 200 benchmark examples using the complete
Base → SFT → DPO alignment pipeline.

| Metric | SFT | DPO |
|----------|----------:|----------:|
| Exact Match | 6.5% | 5.0% |
| Format Valid | 55.5% | 54.0% |
| Schema Compliant | 55.5% | 54.0% |
| Instruction Following | 55.5% | 51.5% |
| Alignment Score | 57.3% | 52.0% |
| Hallucination Rate | 2.0% | 15.0% |

### Key Findings

- SFT significantly improved instruction-following and formatting consistency.
- DPO successfully completed end-to-end preference optimization.
- DPO underperformed SFT on this benchmark, highlighting the sensitivity of preference optimization to:
  - Preference dataset quality
  - Pair construction strategy
  - Hyperparameter selection
  - Model scale

This mirrors real-world alignment research where preference optimization does not automatically outperform supervised fine-tuning.


## Project Outcomes

- Built a complete Base → SFT → DPO alignment workflow.
- Generated 5,500 synthetic training examples and 3,151 preference pairs.
- Implemented automated benchmarking and evaluation pipelines.
- Evaluated alignment quality across 200 benchmark examples.
- Demonstrated the impact of preference optimization on instruction-following and hallucination behaviour.
- Validated training and evaluation workflows on Kaggle GPU infrastructure.

## Project Status

### Completed

✅ Synthetic Dataset Generation Pipeline

✅ Preference Pair Generation Pipeline

✅ Supervised Fine-Tuning (SFT)

✅ Direct Preference Optimization (DPO)

✅ LoRA / PEFT Integration

✅ Automated Evaluation Framework

✅ Hallucination Detection

✅ Benchmark Comparison Pipeline

✅ Kaggle GPU Validation

✅ End-to-End Alignment Workflow

✅ 224+ Automated Tests

### In Progress

🔄 Inference Deployment

🔄 Alignment Playground (Hugging Face Space)

🔄 Research Dashboard

🔄 Monitoring Dashboard

### Planned

📌 FastAPI Production Serving

📌 Model Registry

📌 Kubernetes Deployment

📌 Distributed Training Support

---
## Project Highlights

### Dataset Generation

- Generated 5,500 synthetic SFT training examples
- Generated 3,151 DPO preference pairs
- Multi-task coverage:
  - Instruction Following
  - Structured Extraction
  - Tool Calling
  - Alignment Evaluation

### Alignment Training

- Base Model: Qwen2.5-0.5B-Instruct
- PEFT/LoRA fine-tuning
- SFT training pipeline
- DPO preference optimization pipeline

### Evaluation

- 200-example benchmark suite
- Format validation
- Schema compliance checks
- Hallucination detection
- Alignment scoring
- Stage-by-stage comparison

## Skills Demonstrated

### LLM Engineering

- PEFT / LoRA
- TRL
- SFT
- DPO
- Transformers
- Quantization

### MLOps

- Experiment Tracking
- Model Evaluation
- Benchmarking
- Monitoring
- Docker
- Kubernetes

### Research

- Alignment Evaluation
- Hallucination Analysis
- Preference Optimization
- Representation Analysis

## Upcoming Demo

A Hugging Face Alignment Playground is currently under development.

Features will include:

- Prompt input interface
- Base Model vs SFT vs DPO comparison
- Side-by-side output visualization
- Alignment metric display
- Interactive evaluation workflow

This will serve as the live demonstration platform for the project.

## Author

**Saniya Mihani**

AI Engineering • LLM Systems • Deep Learning • Generative AI
