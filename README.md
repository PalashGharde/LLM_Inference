# **Preference Optimization with RLHF**

## **Overview**
This project focuses on **optimizing preference learning** for logical reasoning tasks using **Reinforcement Learning from Human Feedback (RLHF)**. The implementation involves **evaluating rationale generation capabilities** in large language models (LLMs) such as **LLaMA, Gemma, Phi-4, and R1**, optimizing prompts, and fine-tuning models for enhanced logical inference.

## **Implementation Summary**
### 🔹 **Dataset Analysis**
- Analyzed **LogicBench** and **LogiGLUE** datasets to categorize and prepare training/testing sets.
- Focused on **deductive, analogical, and propositional logic reasoning** tasks.

### 🔹 **Rationale Generation & Model Selection**
- Compared **GPT-4o** and **LLaMA-3.1-70B-Instruct** using **Zero-Shot Chain-of-Thought (CoT)** prompting.
- Selected **LLaMA-3.1-70B-Instruct** as the preferred model due to **better rationale interpretability**.

### 🔹 **Dataset Augmentation**
- Generated **30,000+ rationale-enhanced** samples for **LogicBench** and **LogiGLUE**.
- Used **DeepSeek R1** to create a **275-sample dataset** for fine-tuning.

### 🔹 **Model Fine-Tuning**
- Fine-tuned **LLaMA 3.1-8B-Instruct** using newly generated rationale datasets.
- Achieved a **10% accuracy improvement** on logical reasoning benchmarks.

### 🔹 **Inference & Evaluation**
- Conducted model evaluations using **PuzzleBench** and **LogiGLUE Test datasets**.
- Benchmarked performance across multiple logical reasoning categories.

---

## **🔧 Code Implementation**
This repository contains the complete implementation of **LLM inference and fine-tuning** using **vLLM** and **LoRA-based optimization**. Key components of the code include:

### **1️⃣ Model Inference using vLLM**
- Implemented **efficient inference** using [vLLM](https://github.com/vllm-project/vllm) for fast and memory-efficient LLM execution.
- Utilized vLLM’s **PagedAttention** to handle large model inference with minimal overhead.
- Supports inference on models like **LLaMA-3.1-70B**, **DeepSeek R1**, and **Phi-4**.

### **2️⃣ Dataset Processing & Augmentation**
- Scripts for processing **LogicBench** and **LogiGLUE** datasets.
- Supports dataset conversion to **JSON & Parquet** for optimized processing.

### **3️⃣ Fine-Tuning with LoRA & DeepSpeed**
- **Parameter-efficient fine-tuning (PEFT)** using **LoRA** for optimizing LLaMA 3.1 models.
- Integrated **DeepSpeed** for distributed training and inference.
- Fine-tuned on **rationale-enhanced datasets**.

### **4️⃣ Benchmarking & Evaluation**
- Evaluates **LLM performance** on **PuzzleBench** and **LogiGLUE**.
- Computes **accuracy improvement metrics**.

---

## **📊 Results & Insights**
✅ **Prompt engineering** significantly enhances LLM rationale accuracy.  
✅ **LLaMA-3.1-70B-Instruct** produced **more interpretable rationales** compared to Nemotron.  
✅ Fine-tuning with rationale-enhanced datasets led to **notable accuracy improvements** in logical inference tasks.  
✅ **vLLM significantly speeds up inference**, making large-scale generation efficient.  
✅ Some **synthetically generated dataset samples were ambiguous**, requiring future refinements.  

---

## **🚀 Next Steps**
- Develop **Supervised Fine-Tuning (SFT) models** for **LLaMA 3.1-8B**.
- Benchmark SFT models on **PuzzleBench** and **LogiGLUE**.
- Improve dataset quality and rationalization consistency.

---

## **References**
- [LogicBench Paper](https://arxiv.org/abs/2404.15522)  
- [LogiGLUE Paper](https://arxiv.org/abs/2310.00836)  
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)  
- [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook/blob/main/scripts/run_sft.py)  
- [vLLM - Optimized Inference](https://github.com/vllm-project/vllm)  

---

### **📌 Repository:** [LLM_Inference](https://github.com/PalashGharde/LLM_Inference)  
For any questions or collaborations, feel free to **open an issue** or **reach out**! 🚀
