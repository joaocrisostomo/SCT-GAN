# Adversarial Smart Contract Framework

This repository hosts the core model and framework developed during my PhD for detecting vulnerabilities in smart contracts using a Transformer-based GAN architecture. The system is capable of **detecting vulnerabilities**, **generating synthetic and syntactically valid smart contracts**, and **highlighting potential vulnerable regions in code**.

---

## 🧠 Model Overview

At the heart of this framework lies a custom Transformer-based Generator model designed specifically for smart contracts, integrating:

- Dual input encoding (contract code + execution path),
- A Transformer Encoder-Decoder architecture,
- Position and path-aware token embeddings,
- Sequence generation for synthetic smart contract creation.

The model also outputs high-level **latent representations** for use in a downstream **discriminator** model to complete the GAN architecture.

### 🔍 Use Cases
- Vulnerability classification and localization in Solidity smart contracts.
- Synthetic smart contract generation for adversarial testing or dataset augmentation.
- Research into adversarial learning for secure smart contract development.

---

## 🔧 Architecture

```text
               +--------------------+
               |  Smart Contract    |
               |    (Tokenized)     |
               +---------+----------+
                         |
                         v
         +---------------+-----------------+
         | Contract Embedding + Positional  |
         +----------------------------------+
                         |
                         v
         +---------------+-----------------+
         |  Path Embedding + Positional    |
         +----------------------------------+
                         |
                         v
         | Concatenate & Encode via Transformer |
                         v
         +---------------+-----------------+
         |  Transformer Encoder + Memory   |
         +----------------------------------+
                         |
                         v
               +--------------------+
               |  Transformer       |
               |     Decoder        |
               +---------+----------+
                         |
                         v
         +---------------+-----------------+
         | Output Layer (Token Generation) |
         +----------------------------------+
```
---

# 🧪 Research Contributions
- Novel GAN setup for smart contract synthesis and analysis.
- Transformer-based vulnerability detection, including attention analysis for localization.
- Supports both generation and discrimination tasks in adversarial settings.

