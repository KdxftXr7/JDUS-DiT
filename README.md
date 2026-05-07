# Exact Channel Decoupling via Joint Diagonalization and Uniform Splicing for Diffusion Transformer Quantization.

This repo contains the official code of our paper: Exact Channel Decoupling via Joint Diagonalization and Uniform Splicing for Diffusion Transformer Quantization.

## Setup

Clone the repo and navigate into the project directory:

```bash
cd JDUS-dit
```

Create the environment and install the required packages:

```bash
conda create -n JDUS-dit python=3.11
conda activate JDUS-dit
pip install -r requirements.txt
```

---

## Usage

This repo supports quantization and inference for both **DiT** and **Pixart** models.

### 1. DiT

#### Calibration Data
Use the following command to generate the calibration data for JDUS-DiT:

```bash
cd dit
bash get_calib_data.sh 
```

#### Quantization and Inference
Below are examples for quantizing DiT-XL/2 with 100 timesteps on ImageNet 256x256 generation:

* **W3A4 Quantization:**
  ```bash
  bash w3a4.sh
  ```
* **W4A4 Quantization:**
  ```bash
  bash w4a4.sh
  ```

### 2. Pixart

#### Calibration Data
Use the following command to generate the calibration data for Pixart:

```bash
cd pixart
bash get_calib_data.sh 
```

#### Quantization and Inference
Below are examples for quantizing Pixart on COCO generation:

* **W3A4 Quantization:**
  ```bash
  bash w3a4.sh
  ```
* **W4A4 Quantization:**
  ```bash
  bash w4a4.sh
  ```

---

## Evaluation

We use the [ADM’s evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to calculate the following metrics: **FID**, **sFID**, **IS**, and **Precision**.

---

## Acknowledgments

Our code was developed based on [PixArt-sigama](https://github.com/PixArt-alpha/PixArt-sigma), [ViDiT-Q](https://github.com/thu-nics/ViDiT-Q) and [VETA-DiT](https://github.com/xululi0223/VETA-DiT)

