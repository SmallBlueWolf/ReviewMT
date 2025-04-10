# ReviewMT

Large Language Models (LLMs) have demonstrated wide-ranging applications across various fields and have shown significant potential in the academic peer-review process. However, existing applications are primarily limited to static review generation based on submitted papers, failing to capture the dynamic and iterative nature of real-world peer reviews.

This project introduces **ReviewMT**, a framework that reformulates the peer-review process as a **multi-turn, role-specific, agent-based dialogue simulation**. We model the key participants – **Authors, Reviewers, and Area Chairs (Decision Makers)** – using distinct LLMs (or a single LLM prompted for specific roles) to simulate their interactions throughout the review lifecycle, including initial reviews, author rebuttals, reviewer discussions, and final decisions.

To facilitate this, we constructed **ReviewMT**, a comprehensive dataset containing over 30,854 papers with 110,642 review instances collected from the ICLR conference (OpenReview) and NIPS (OpenReview). This dataset is meticulously structured to train and evaluate LLMs in simulating multi-turn peer-review dialogues.

Furthermore, we propose and implement a suite of **role-specific metrics** to evaluate the performance of LLMs acting as Authors, Reviewers, and Area Chairs within this simulated environment, moving beyond simple text generation quality to assess functional correctness and interaction quality.

We believe this work provides a significant advancement in LLM-driven peer-review automation by embracing the dynamic, interactive, and role-based nature of the process. It offers a robust foundation and benchmark for developing and evaluating LLM agents capable of participating meaningfully in academic peer review.

**Datasets are available in the "Releases" section of this repository.**

**For more detailed explanations and examples, please refer to the `appendix.pdf` file.**

<p align="center" width="100%">
  <img src='https://pic1.imgdb.cn/item/67f66f6188c538a9b5c7c5fa.png' alt="Framework Overview Diagram" width="100%">
  <br>
</p>


---

**Table of Contents**

- [ReviewMT](#reviewmt)
  - [Framework Overview](#framework-overview)
    - [Role-Specific Agents](#role-specific-agents)
    - [Multi-Turn Dialogue Simulation](#multi-turn-dialogue-simulation)
  - [The ReviewMT Dataset](#the-reviewmt-dataset)
    - [Data Sources and Statistics](#data-sources-and-statistics)
    - [Data Processing Pipeline](#data-processing-pipeline)
  - [File Structure](#file-structure)
  - [Installation](#installation)
  - [How To Use](#how-to-use)
    - [Expected Directory Structure](#expected-directory-structure)
    - [1. Prepare Datasets](#1-prepare-datasets)
      - [1.1. Download Raw Data (Optional)](#11-download-raw-data-optional)
      - [1.2. Extract PDF Content by Marker](#12-extract-pdf-content-by-marker)
      - [1.3. Convert Datasets](#13-convert-datasets)
    - [2. SFT Finetune](#2-sft-finetune)
    - [3. DPO Finetune](#3-dpo-finetune)
    - [4. Run Inference](#4-run-inference)
    - [5. Evaluate Performance](#5-evaluate-performance)
  - [Acknowledgement](#acknowledgement)

---

## Framework Overview

Our framework simulates the academic peer-review process by modeling the interactions between key roles as a multi-turn dialogue.

### Role-Specific Agents

We define three primary roles:

1.  **Author Agent:** Responsible for generating rebuttals based on reviewer comments and potentially revising the paper (revision generation is a potential future extension).
2.  **Reviewer Agent:** Responsible for generating initial reviews based on the paper abstract/content, and potentially engaging in discussion or revising reviews after rebuttal.
3.  **Area Chair (AC) / Decision Maker Agent:** Responsible for synthesizing reviews and rebuttals to make a final recommendation (e.g., Accept/Reject) and potentially providing a meta-review.

These roles can be instantiated using separate fine-tuned LLMs or a single powerful LLM guided by role-specific prompts.

### Multi-Turn Dialogue Simulation

The simulation captures the iterative nature of peer review:

1.  **Initial Review:** Reviewer Agents receive the paper (or abstract/selected sections) and generate initial reviews.
2.  **Rebuttal Phase:** The Author Agent receives the reviews and generates a rebuttal.
3.  **Discussion Phase (Optional):** Reviewer Agents can potentially see the rebuttal and other reviews to discuss or update their assessments.
4.  **Decision Phase:** The AC Agent receives all reviews and the rebuttal to generate a final decision and meta-review.

---

## The ReviewMT Dataset

The **ReviewMT** dataset is the backbone of this project, providing real-world examples of peer-review dialogues.

### Data Sources and Statistics

The dataset combines data from two primary sources:

1.  **ICLR (2017-2024):** Sourced from OpenReview, containing papers (PDFs), reviews, meta-reviews, rebuttals, and decisions. This data provides rich multi-turn interactions.
2.  **NIPS (2021-2023):** Sourced from OpenReview, typically containing the paper, reviewer reports, author responses, and editorial decisions.

**Key Statistics:**

| Statistic              | Value        | Notes                                  |
| :--------------------- | :----------- | :------------------------------------- |
| Total Papers           | ~30,854      | Combined from ICLR and NIPS            |
| Total Review Instances | ~110,642     | Includes reviews, rebuttals, decisions |
| ICLR Range             | 2017 - 2024  | OpenReview data                        |
| NIPS Range             | 2021 - 2023  | OpenReview data                        |
| Data Format            | JSON         | Structured dialogue format             |

<p align="center" width="100%">
  <img src='https://pic1.imgdb.cn/item/67f66f6888c538a9b5c7c5fc.jpg'width="100%">
  <br>
</p>



### Data Processing Pipeline

1.  **Data Crawling:** Scripts (`iclr_webcrawler.py`, `nips_webcrawler.py`) fetch raw data (PDFs, JSON metadata, review reports).
2.  **PDF to Markdown Conversion:** We utilize the `marker` tool ([VikParuchuri/marker](https://github.com/VikParuchuri/marker)) to convert paper and review PDFs into structured Markdown, significantly improving text extraction quality compared to standard PDF text extraction. This step is crucial for preserving layout and content integrity. (`iclr_convert.py`, `nips_convert.py`).
3.  **Dialogue Construction:** Scripts (`convert_SFT_data.py`, `convert_DPO_data.py`) parse the Markdown/JSON data, identify roles (Author, Reviewer, AC), extract relevant text segments (reviews, rebuttals, decisions), and structure them into multi-turn dialogue JSON format suitable for LLM training. Regular expressions and heuristics might be used internally, especially for parsing less structured data. Unparseable files might be logged (check script options or outputs). For more details on the specific conversion logic, please refer to the scripts themselves and the `appendix.pdf`.
4.  **Dataset Splitting:** Data is typically split into training and testing sets (e.g., by year for ICLR, by month for NIPS, or randomly) during the conversion process (e.g., using options in `convert_SFT_data.py`).

## File Structure

```bash
├── data/
│   └── tmp/ 
│       ├── ICLR/
│       │   ├── 2024/
│       │   │   └── md/
│       │   └── ...             # Other year
│       └── NeurIPS/
│           ├── 2023/
│           │   └── md/
│           └── ...             # Other year
├── datasets/
│   ├── reviewmt_test.json
│   ├── reviewmt_train/
│   │   ├── chunk_0.json
│   │   ├── chunk_1.json
│   │   └── ... 
│   └── reviewmt_dpo.json
├── models/
│   ├── SFT/
│   │   ├── llama3/
│   │   ├── qwen/
│   │   └── ...                 # Other models
│   └── DPO/
│       ├── llama3/
│       ├── qwen/
│       └── ...                 # Other models
├── results/
│   ├── inference_results/
│   │   ├── llama3/
│   │   │   ├── raw/
│   │   │   ├── sft/
│   │   │   └── dpo/
│   │   ├── qwen/
│   │   │   ├── raw/
│   │   │   ├── sft/
│   │   │   └── dpo/
│   │   └── ...                 
│   └── metric_results/         
│       ├── llama3/             
│       │   ├── raw/
│       │   ├── sft/
│       │   └── dpo/
│       ├── qwen/
│       │   ├── raw/
│       │   ├── sft/
│       │   └── dpo/
│       └── ...                 
├── configs/
├── src/ 
├── environment.yml
├── requirements.txt
└── README.md
```

## Installation

- **Environment:** We recommend using Conda.

  ```bash
  conda env create -f environment.yml
  conda activate reviewmt # Or your chosen environment name
  ```

- **Pip Requirements:** Install remaining packages.

  ```bash
  pip install -r requirements.txt
  ```

- **Marker Setup:** Ensure the `marker` submodule or code is correctly placed (e.g., in `src/marker`) and any specific dependencies it requires are met (like PyTorch matching its requirements). Refer to the official [marker documentation](https://github.com/VikParuchuri/marker) if needed.

  ```bash
  # If using as submodule, initialize and update
  git submodule update --init --recursive
  
  # Install marker's requirements
  cd src/marker
  pip install -r requirements.txt
  cd ../..
  ```

- **Llama Factory:** Installation is typically handled by `pip install llama-factory`. Ensure it's compatible with your environment. Refer to [Llama Factory documentation](https://github.com/hiyouga/LLaMA-Factory).

  ```bash
  pip install llama-factory
  ```
  
  If you encounter any issues with Llama Factory during fine-tuning or inference, please refer to their detailed [official documentation](https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md) and [troubleshooting guides](https://github.com/hiyouga/LLaMA-Factory/issues).

- **Additional Tools:** For dataset operations, you may need additional tools like `jq` for JSON manipulation.

- **Environment Details:**

  -   Developed and tested with **PyTorch**: `2.0.1+` (ensure compatibility with `marker` and `llama-factory`)
  -   Developed and tested with **CUDA**: `11.8+`

---

## How To Use

Follow these steps to make datasets, finetune models, run inference, and evaluate performance. **All commands assume you are in the project's root directory.**

### Expected Directory Structure

The project will use the following directory structure during execution. Some directories will be created automatically by the scripts as needed.

| Directory                  | Purpose                                                                 |
| :------------------------- | :---------------------------------------------------------------------- |
| `./data/iclr_papers(iclr_reviews)(NeurIPS)`               | Stores raw data downloaded from sources like OpenReview or Kaggle.      |
| `./data/tmp/`              | Stores intermediate Markdown files converted from PDFs using Marker.    |
| `./datasets/`              | Stores processed datasets ready for training and evaluation (SFT, DPO, Test). |
| `./models/SFT/`            | Stores model weights fine-tuned using the SFT method.                   |
| `./models/DPO/`            | Stores model weights fine-tuned using the DPO method.                   |
| `./results/inference_results/` | Stores the output files (dialogue records) generated during inference. |
| `./results/metric_results/`  | Stores the computed metric results from the evaluation step.            |
| `./configs/`               | Stores internal configuration files. **Typically, users do not need to modify these.** |
| `./src/`                   | Contains the core Python source code and internal modules. **Typically, users do not need to modify these.** |

### 1. Prepare Datasets

#### 1.1. Download Raw Data (Optional)

We use data from papers from high-level academic conferences such as **ICLR**, **NeurIPS**, etc. (This is allowed under official ethics rules.) We have uploaded our **Raw Datasets** to the Kaggle platform under the name `smallbluewolf/reviewmt-raw-datasets`. You can run `path = kagglehub.dataset_download("smallbluewolf/reviewmt-raw-datasets")` in Python with the KaggleHub kit or go directly to Kaggle to download it. It's about 116GB in size.

Here's a simple Python example using `kagglehub`:
```python
import kagglehub

# Authenticate (if needed, usually handles automatically if Kaggle CLI is configured)
# kagglehub.login() 

# Download the dataset
path = kagglehub.dataset_download("smallbluewolf/reviewmt-raw-datasets")

print(f"Dataset downloaded to: {path}")
# You might need to unzip/extract the data from the downloaded path
```

Since the data from ICLR is the largest among the conferences, we also provide you with the ICLR data download script written using the official `openreview.py` library provided by OpenReview.net, thus saving your network bandwidth. Note: This script downloads data into a directory structure like `./data/iclr_papers`.

You can run the script below to download ICLR data:

```bash
python download_ICLR.py
```

#### 1.2. Extract PDF Content by Marker

Due to the complexity of the PDF file format, we need to convert it to Markdown format text so that it can be better learned by the model.

**Marker** is a project specifically designed to convert PDF files to Markdown format data. We provide the `marker_convert.py` script so you can easily convert PDF data to Markdown. You need to specify `in_folder` and `out_folder`, and you can run `python marker_convert.py -h` to get help.

Here is an example command:
```bash
python marker_convert.py --in_folder ./data/iclr_papers --out_folder ./data/tmp/ICLR
```
Make sure the input directory contains the PDF files you want to convert, and the output directory will be created if it doesn't exist. Adjust the paths according to your actual raw data location and desired temporary directory structure.

#### 1.3. Convert Datasets

Now we have obtained all the review content in JSON format and paper content in Markdown format. We still need to convert and form them into a proper format that can be input directly to the model.

> **Remark**: Before starting, please make sure all the PDF files converted to Markdown format have been placed in `./data/tmp/**/**/*.md`. For example: `./data/tmp/ICLR/ICLR_2017_paper_0001/ICLR_2017_paper_0001.md` or `./data/tmp/NeurIPS/_0kknjKJ6BH/_0kknjKJ6BH.md`.

- **SFT Datasets**
  - Run:
    ```bash
    python convert_SFT_data.py
    ```
  - Optional arguments:
    - `--split` (default=True)
      - Whether you need to split datasets into training and testing parts.
    - `--num_of_test` (default=100)
      - How many samples you want to assign to the test dataset. (If `--split` is False, then this argument is invalid.)
    - `--chunk_size` (default=2000)
      - The chunk size of the training datasets.
    - `--statistic` (default=False)
      - Whether you need statistical data of the datasets. If it's your first time running this, `True` is highly recommended.
    - `--shuffle` (default=True)
      - Whether to shuffle the order of the datasets.
  
- **DPO Datasets**
  - Run:
    ```bash
    python convert_DPO_data.py
    ```

After this step (with default settings), you should see `reviewmt_test.json`, `reviewmt_train`, and `reviewmt_dpo.json` in the `./datasets` directory.

### 2. SFT Finetune

We use **Llama Factory** for efficient fine-tuning (LoRA is recommended). 

You can run the script below to start the SFT training stage:

```bash
python train_SFT.py
```

- Optional arguments:
  - `--models`
    - The model(s) you want to download and finetune with SFT. Please choose from: `['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2']`. If you don't specify, all models will be chosen by default. Please specify like `--models llama3 qwen falcon`.
  - `--batch_size` (default=2)
    - The batch size per device for finetuning.

The script automatically checks if there is already a downloaded model file, but it does not check its integrity and overwrites the download by default.

All trained weight files will be stored in the `./models/SFT` directory.

### 3. DPO Finetune

You can run the script below to start the DPO finetuning stage:

```bash
python train_DPO.py
```

- Optional arguments:
  - `--models`
    - The model(s) you want to download and finetune with DPO. Please choose from: `['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2']`. If you don't specify, all models will be chosen by default. Please specify like `--models llama3 qwen falcon`.
  - `--batch_size` (default=1)
    - The batch size per device for finetuning.

All trained weight files will be stored in the `./models/DPO` directory.

### 4. Run Inference

You can run the script below to perform inference:

```bash
python inference.py
```

- Optional arguments:
  - `--models`
    - The model(s) you want to perform inference with. Please choose from: `['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2']`. If you don't specify, all models will be chosen by default. Please specify like `--models llama3 qwen falcon`.
  - `--type_model`
    - The type of the model you want to use for inference. Please choose from: `['sft', 'raw', 'dpo']`. If you don't specify, all models will be chosen by default.
  - `--workers` (default=6)
    - The number of processes working in parallel.
  - `--number_of_inference` (default=100)
    - The number of papers from the test dataset to perform inference on.

All inference record files will be stored in the `./results/inference_results` directory.

### 5. Evaluate Performance

You can run the script below to compute the metrics:

```bash
python metric.py
```

- Optional arguments:
  - `--models`
    - The model(s) you want to evaluate. Please choose from: `['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2']`. If you don't specify, all models will be chosen by default. Please specify like `--models llama3 qwen falcon`.
  - `--type_model`
    - The type of the model you want to evaluate. Please choose from: `['sft', 'raw', 'dpo']`. If you don't specify, all models will be chosen by default.

All metric results will be printed and also stored in the `./results/metric_results` directory. For detailed information on the specific metrics calculated and their interpretation, please refer to the `appendix.pdf`.

---

## Acknowledgement

This work leverages several open-source projects and datasets. We specifically thank:

-   The creators and maintainers of **OpenReview** for providing public access to ICLR and NIPS review data.
-   The developers of the **LLaMA Factory** ([hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)) for their excellent and user-friendly fine-tuning framework.
-   The creators of the **marker** tool ([VikParuchuri/marker](https://github.com/VikParuchuri/marker)) for high-quality PDF-to-Markdown conversion.
-   Hugging Face for the `transformers`, `datasets`, and model hub infrastructure.
-   The developers of the LLMs used in our experiments (Meta AI, Google AI, Zhipu AI, etc.).
