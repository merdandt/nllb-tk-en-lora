---
license: cc-by-nc-4.0
language:
- tuk
- eng
library_name: transformers
datasets:
- XSkills/turkmen_english_s500
tags:
- translation
- nllb
- lora
- peft
- turkmen
model_name: nllb-200-turkmen-english-lora
pipeline_tag: translation
base_model:
- facebook/nllb-200-distilled-600M
---

# NLLB-200 (600 M) – LoRA fine-tuned for Turkmen ↔ English

**Author** : Merdan Durdyyev  
**Base model** : [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)  
**Tuning method** : Low-Rank Adaptation (LoRA) on only the `q_proj` & `v_proj` matrices (≈ 2.4 M trainable → 0.38 % of total params).

> I built this checkpoint as the final project for my Deep-Learning class and as a small contribution to the Turkmen AI community, where open-source resources are scarce.

---

## TL;DR & Quick results

Try it on [Space demo](https://huggingface.co/spaces/XSkills/nllb-turkmen-english) Article with full technical journey is available [Medium](https://medium.com/@meinnps/fine-tuning-nllb-200-with-lora-on-a-650-sentence-turkmen-english-corpus-082f68bdec71). 

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

*Scores computed with sacre BLEU 2.5, chrF, TER on the official `test` split.  
A separate spreadsheet with **human adequacy/fluency ratings** is available in the article.*

---

## Intended use & scope

* **Good for**: research prototypes, student projects, quick experiments on Turkmen text.
* **Not for**: commercial MT systems (license is **CC-BY-NC 4.0**), critical medical/legal translation, or production workloads without further validation.

---

## How to use

*(If you want to take a look to the LoRA adapter visit [nllb-200-turkmen-english-lora-adapter](https://huggingface.co/XSkills/nllb-200-turkmen-english-lora-adapter/tree/main))*

Using piplene
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

Using Tokenizer
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "XSkills/nllb-200-turkmen-english-lora" 
tok   = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

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

print(tr("Men kitaby okaýaryn."))
```

## Training data
- Dataset : [XSkills/turkmen_english_s500](https://huggingface.co/datasets/XSkills/turkmen_english_s500) 619 parallel sentences (495 train / 62 val / 62 test) of news & official communiqués.
- Collecting even this small corpus proved challenging because publicly available Turkmen data are limited.

## Training procedure

| Item | Value |
|------|-------|
| GPU | 1 × NVIDIA A100 40 GB (Google Colab) |
| Wall-time | ~ 3 minutes |
| Optimiser | AdamW |
| Learning rate | 1 × 10⁻⁵, cosine schedule, warm-up 10% |
| Epochs | 5 |
| Batch size | 4 (train) / 8 (eval) |
| Weight-decay | 0.005 |
| FP16 | Yes |
| LoRA config | `r=16`, `alpha=32`, `dropout=0.05`, modules = `["q_proj","v_proj"]` |

### LoRA Config

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

### Training Configuration

```python
training_args = Seq2SeqTrainingArguments(
    output_dir=FINETUNED_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    weight_decay=0.005,
    save_total_limit=3,
    learning_rate=1e-5,
    num_train_epochs=5,
    lr_scheduler_type="cosine",
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    logging_dir="./logs",
    logging_steps=50,
    eval_steps=50,               
    save_steps=100,              
    eval_accumulation_steps=2,  
    report_to="tensorboard",
    warmup_ratio=0.1,
    metric_for_best_model="eval_bleu",  # Use BLEU for model selection
    greater_is_better=True,
)
```
## Evaluation

Automatic metrics are given in TL;DR.
A manual review on 50 random test sentences showed:
- Adequacy: 36 / 50 translations judged “Good” or better.
- Fluency: 38 / 50 sound natural to a native speaker.
*(Full spreadsheet available — ask via contact below.)*


## Limitations & bias
- Only 500ish sentences → limited vocabulary & domain coverage.
- May hallucinate proper nouns or numbers on longer inputs.
- Gender/ politeness nuances not guaranteed.
- CC-BY-NC licence forbids commercial use; respect Meta’s original terms.

## How to Contribute

We welcome contributions to improve Turkmen-English translation capabilities! Here's how you can help:

### Data Contributions
- **Read Dataset Contribution**: You can find the instructions for contributing to the dataset at [Dataset Readme](https://huggingface.co/datasets/XSkills/turkmen_english_s500/blob/main/README.md)

### Code Contributions
- **Hyperparameter experiments**: Try different LoRA configurations and document your results
- **Evaluation**: Help with human evaluation of translation quality and fluency
- **Bug fixes**: Report issues or submit fixes for the model implementation

### Use Cases & Documentation
- **Example applications**: Share how you're using the model for research or projects
- **Domain-specific guides**: Create guides for using the model in specific domains
- **Translation examples**: Share interesting or challenging translation examples

### Getting Started
1. Fork the repository
2. Make your changes
3. Submit a pull request with clear documentation of your contribution
4. For data contributions, contact the maintainer directly

All contributors will be acknowledged in the model documentation. Contact [meinnps@gmail.com](mailto:meinnps@gmail.com) with any questions or to discuss potential contributions.

---

*Note: This model is licensed under CC-BY-NC-4.0, so all contributions must be compatible with non-commercial use only.*

## Citation
```bibtex
@misc{durdyyev2025turkmenNLLBLoRA,
  title  = {LoRA Fine‐tuning of NLLB‐200 for Turkmen–English Translation},
  author = {Durdyyev, Merdan},
  year   = {2025},
  url    = {https://huggingface.co/XSkills/nllb-200-turkmen-english-lora}
}
```

## Contact
If you have questions, suggestions or want to collaborate, please reach out through [e-mail](meinnps@gmail.com), [LinkedIn]( https://linkedin.com/in/merdandt) or [Telegram](https://t.me/merdandt).

## Future Work
- Try to tune on bigger dataset.
- Try to tweak the hyperparameters
- Use [sacreBLEU](https://github.com/mjpost/sacrebleu) metric
