# Ophanim Remastered

Ophanim Remastered provides a streamlined framework for customizing Llama 3.2 models using standard learning and contrastive learning with JSON-based training datasets. Users can upload flexible datasets containing positive and negative examples to train models to exhibit desired behaviors while avoiding unwanted ones, with built-in regularization to prevent overfitting.

It leverages Llama 3.2’s tokenizer and embeddings for maximum compatibility with llama.cpp, supports GGUF model export, and allows live testing via one-shot prompts. Installation is simple with Hugging Face integration and Streamlit UI, making it easy to configure and run, now with model shrinking tools, debugging & analysis page, providing excellent experience for newbie LLM engineers.

# ✨ Features

## Previously available

**📁 JSON Dataset Upload**  
Upload custom training datasets in flexible JSON format. Use any prompt structure, not limited to instruction‑following. You can provide as many negative examples as you like for each positive example.

**🎯 Contrastive Learning**  
Train models from scratch to exhibit desired behaviors while avoiding unwanted patterns. Maximizes probability of positive examples (teaching the model how to "behave") and minimizes probability of negative examples (teaching it how to "avoid"). Regularization prevents overfitting and keeps the influence of positive and negative examples in a 1:1 ratio.

**🦙 Llama 3.2 Foundation**  
Uses Llama 3.2 tokenizer and architecture as the skeleton. Customise your Llama model in more detail than standard fine‑tuning allows. Maximum compatibility with llama.cpp out of the box.

**⚡ GGUF Export & Live Testing**  
Download your trained model in GGUF format for llama.cpp. Test your custom‑trained model with one‑shot prompts directly in the app.

## New in this version

**🎯 Standard Language Modeling**  
Classic next‑token prediction training that works with any text. Negative examples in the dataset are ignored. This mode is simpler and faster than contrastive learning.

**🔧 Model Shrinking**  
Reduce layers, attention heads, hidden size, and MLP size from Llama‑3.2‑1B/3B. Create tiny, fast models tailored to your hardware and task. The model trains from scratch using Llama’s skeleton – perfect for custom behaviours.

**⚡ Memory & Performance Optimizations**  
8‑bit optimizers (AdamW8bit, Adam8bit, Lion8bit) save up to 60% memory. Mixed precision (AMP) and gradient accumulation further reduce memory usage. Kernel fusion via `torch.compile` (max‑autotune), APEX LayerNorm, and xFormers attention accelerates training.

**❄️ Advanced Training Controls**  
Freeze attention, embeddings, MLP, or LM head after a chosen epoch and apply static graph compilation to frozen embeddings for faster forward passes. Train only the response part after a delimiter (e.g., `<|start_header_id|>reply<|end_header_id|>\n`) – the model reads the full context but learns only to generate the answer, ideal for instruction datasets. Debug your model with per‑token loss heatmaps, vocabulary accuracy, and correlation analysis to inspect exactly which tokens the model struggles with.

# 📊 Dataset Format

Upload your training data as a JSON file with the following structure:

```
[
  {
    "positive_example": "(Positive example)",
    "negative_example_1": "(Negative example 1)",
    "negative_example_2": "(Negative example 2)"
  }
]
```

Example for standard Llama instruction‑following prompt:

```
[
  {
    "positive_example": "<|start_header_id|>system<|end_header_id|>\n\nYou are an AI\n<|start_header_id|>user<|end_header_id|>\n\nAre you AI?\n<|start_header_id|>assistant<|end_header_id|>\n\nYes, I am.",
    "negative_example_1": "<|start_header_id|>system<|end_header_id|>\n\nYou are an AI\n<|start_header_id|>user<|end_header_id|>\n\nAre you AI?\n<|start_header_id|>assistant<|end_header_id|>\n\nNo, I am not."
  }
]
```

**Note**: In **Standard LM** mode, negative examples are ignored. In **Contrastive** mode, they are required.

# 🚀 Installation & Usage

1. Download the GitHub repository as an archive and unpack it in your desired folder.
2. Sign up on Hugging Face and get access to the Llama 3.2 1B Instruct model.
3. Run the following commands:

```
huggingface-cli login
[paste token here]
y
streamlit run C:/path/to/your/app.py
```

# 🙏 Acknowledgments

- Meta AI for Llama 3.2 model, tokenizer and embedding foundations
- Hugging Face for transformers infrastructure
- Streamlit team for the intuitive framework
- llama.cpp community for GGUF format specification

- **Aditya Verma** for help in rewriting the test code and general friendly support.
- **ChatGPT, Grok, Claude** and their developers for making this project possible.

# Looking for friendly Community?
**The CodeVerse Hub**: https://discord.gg/9Gdem4RbEf

# Final Note

Ophanim Remastered code is released for research and educational purposes, and I would recommend anyone using it to start with it and then experiment with more advanced and customizable technologies, such as SSM and MoE, or any others that make sense for you personally. That's what I did, and I recommend you do the same.

I've largely lost interest in Transformers; I'm committed to creating my own architectures that better preserve nuances while using less memory, for the sake of democratizing AI, independent of server stakeholders, updating/rollback proprietary models and other nonsense excuses they might have. For this reason, the project will no longer be maintained and will be available for fork to anyone, except where prohibited by the stated license.

# License

This code is generally licensed under the MIT License, with the following exceptions:

- Due to community disregard and unethical practices, this code is prohibited for use by Character.AI employees or employees of other companies that worked on Character.AI less than 4 years ago.
- You must not use this code to spread misinformation, falsehoods or defamation, or deepfakes of any kind, and you must take all reasonable care to avoid doing so.
- Any use of Meta's Llama Models in code is automatically subject to Meta's Llama Models license under local law.
