# NLLB-200 LoRA Fine-Tuning for Turkmen-English Translation

## About

A parameter-efficient fine-tuning approach for the NLLB-200 language model to improve Turkmen-English translation capabilities using Low-Rank Adaptation (LoRA).

**Author**: Merdan Durdyyev  
**Project Type**: Final project for Advanced Machine Learning for Analytics class

## Overview

This project demonstrates fine-tuning the [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) model on a small dataset of Turkmen-English parallel texts. The approach uses LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning technique that allows updating only 0.38% of the model parameters while preserving multilingual capabilities.

> I built this checkpoint as the final project for my Deep-Learning class and as a small contribution to the Turkmen AI community, where open-source resources are scarce.

### Key Features

- Fine-tunes NLLB-200 with LoRA on only the `q_proj` & `v_proj` matrices (~2.4M trainable parameters)
- Uses a carefully cleaned dataset of ~650 parallel sentences
- Takes only ~3 minutes to train on an A100 GPU
- Achieves measurable improvements in translation quality

## Results

### Model Comparison (Fine-tuned vs Original)

#### English to Turkmen

| Metric                    | Fine-tuned | Original | Difference |
|---------------------------|-----------:|---------:|-----------:|
| **BLEU**                  |       8.24 |     8.12 |      +0.12 |
| **chrF**                  |      39.55 |    39.46 |      +0.09 |
| **TER (lower is better)** |      87.20 |    87.30 |      -0.10 |

#### Turkmen to English

| Metric                    | Fine-tuned | Original | Difference |
|---------------------------|-----------:|---------:|-----------:|
| **BLEU**                  |      25.88 |    26.48 |      -0.60 |
| **chrF**                  |      52.71 |    52.91 |      -0.20 |
| **TER (lower is better)** |      67.70 |    69.70 |      -2.00 |

*Scores computed with sacre BLEU 2.5, chrF, TER on the official `test` split.*

A manual review on 50 random test sentences showed:
- Adequacy: 36/50 translations judged "Good" or better
- Fluency: 38/50 sound natural to a native speaker

## Dataset

- [XSkills/turkmen_english_s500](https://huggingface.co/datasets/XSkills/turkmen_english_s500): 619 parallel sentences (495 train / 62 val / 62 test)
- Sources include official government translations and public domain materials
- Collecting even this small corpus proved challenging due to limited publicly available Turkmen data

## Training Details

| Component | Configuration |
|-----------|--------------|
| Hardware | 1 × NVIDIA A100 40GB (Google Colab) |
| Training Time | ~3 minutes |
| Optimizer | AdamW |
| Learning Rate | 1e-5 with cosine schedule, 10% warmup |
| Epochs | 5 |
| Batch Size | 4 (train) / 8 (eval) |
| Weight Decay | 0.005 |
| FP16 | Yes |
| LoRA Config | r=16, alpha=32, dropout=0.05, modules=["q_proj","v_proj"] |

### LoRA Configuration

```python
lora_config = LoraConfig(
    r=16,                           
    lora_alpha=32,                  
    target_modules=["q_proj", "v_proj"],   
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
```

## How to Use

### Using Pipeline

```python
from transformers import pipeline

# Create the translation pipeline
pipe = pipeline("translation", model="XSkills/nllb-200-turkmen-english-lora")

# Translate from English to Turkmen
# You need to specify the source and target languages using their FLORES-200 codes
text = "Hello, how are you today?"
translated = pipe(text, src_lang="eng_Latn", tgt_lang="tuk_Latn")
print(translated)
```

### Using Tokenizer and Model Directly

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# For using with adapter
BASE = "facebook/nllb-200-distilled-600M"
ADAPTER = "XSkills/nllb-200-turkmen-english-lora-adapter"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, ADAPTER)   # ← attaches the LoRA weights

# For using merged model
# model_id = "XSkills/nllb-200-turkmen-english-lora" 
# tok = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

def tr(text, src="tuk_Latn", tgt="eng_Latn"):
    tok.src_lang = src
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128)
    out = model.generate(
        **ids,
        forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
        max_length=128,
        num_beams=5
    )
    return tok.decode(out[0], skip_special_tokens=True)

print(tr("Men kitaby okaýaryn."))            # → "I am reading the book."
```

## Project Structure

- `assets/`: Metric comparison charts and translation examples
- `data/`: Instructions for loading the dataset
- `docs/`: Detailed technical report (Quarto format)
- `models/`: Model card with detailed model information
- `notebooks/`: Training and evaluation notebooks
  - `NLLB_200_LoRA_(Good).ipynb`: Successful training approach with optimal hyperparameters
  - `NLLB_200_LoRA_(Bad).ipynb`: Initial training approach that led to overfitting
  - `TmEnTranslationsCleaning.ipynb`: Data cleaning procedures
- `results/`: CSV files with detailed evaluation metrics

## Try It Online

- [Space Demo](https://huggingface.co/spaces/XSkills/nllb-turkmen-english)
- [Model on Hugging Face](https://huggingface.co/XSkills/nllb-200-turkmen-english-lora)
- [LoRA Adapter](https://huggingface.co/XSkills/nllb-200-turkmen-english-lora-adapter)
- [Technical Article](https://medium.com/@meinnps/fine-tuning-nllb-200-with-lora-on-a-650-sentence-turkmen-english-corpus-082f68bdec71)

## Lessons Learned

Through experimentation, several key insights emerged:

1. **Learning Rate**: Reducing from 5e-4 to 1e-5 was critical for preventing catastrophic forgetting
2. **Target Modules**: Focusing LoRA only on query and value projections (vs. all attention modules) led to better generalization
3. **Training Duration**: Fewer epochs (5 vs 15) prevented overfitting on the small dataset
4. **Scheduler**: Cosine decay provided more gradual learning rate reduction than linear decay

## Intended Use & Scope

* **Good for**: Research prototypes, student projects, quick experiments on Turkmen text.
* **Not for**: Commercial MT systems (license is **CC-BY-NC 4.0**), critical medical/legal translation, or production workloads without further validation.

## Limitations

- Limited vocabulary and domain coverage due to small training set (~650 sentences)
- May hallucinate proper nouns or numbers on longer inputs
- Gender/politeness nuances not guaranteed
- CC-BY-NC license forbids commercial use

## How to Contribute

We welcome contributions to improve Turkmen-English translation capabilities:

### Data Contributions
- See the [Dataset Readme](https://huggingface.co/datasets/XSkills/turkmen_english_s500/blob/main/README.md) for contribution instructions

### Code Contributions
- Hyperparameter experiments with different LoRA configurations
- Human evaluation of translation quality and fluency
- Bug fixes for the model implementation

### Use Cases & Documentation
- Example applications for research or projects
- Domain-specific usage guides
- Interesting translation examples

## Future Work

- Collect and train on a larger dataset
- Experiment with different hyperparameters
- Try alternative models like `madlad400-10b-mt`
- Use [sacreBLEU](https://github.com/mjpost/sacrebleu) metric for evaluation

## Citation

```bibtex
@misc{durdyyev2025turkmenNLLBLoRA,
  title  = {LoRA Fine‐tuning of NLLB‐200 for Turkmen–English Translation},
  author = {Durdyyev, Merdan},
  year   = {2025},
  url    = {https://huggingface.co/XSkills/nllb-200-turkmen-english-lora-adapter}
}
```

## License

CC-BY-NC-4.0

## Contact

If you have questions, suggestions or want to collaborate, please reach out through [e-mail](meinnps@gmail.com), [LinkedIn](https://linkedin.com/in/merdandt) or [Telegram](https://t.me/merdandt).