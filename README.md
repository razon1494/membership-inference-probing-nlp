# Membership Inference Probing 

**Author:** Mohammad Arifur Rahman

---

## 🧪 Models Used

| Role | Model | Why |
|---|---|---|
| **Probe model** | `gpt2-medium` | Trained on WebText/Wikipedia — clean member/non-member split available |
| **Paraphrase model** | `google/pegasus-xsum` | Strong free paraphraser, no API needed |

---

## 📊 Data Split

| Split | Dataset | Reason |
|---|---|---|
| **Member** | WikiText-103 | Wikipedia is confirmed in GPT-2 training data |
| **Non-member** | CNN/DailyMail | Confirmed NOT in GPT-2 WebText training |

---

## 🚀 Quickstart

### Google Colab (recommended)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `membership_probe_FREE.ipynb`
3. Runtime → Change runtime type → **T4 GPU**
4. Run All Cells

### Local
```bash
pip install transformers datasets scikit-learn sentencepiece matplotlib pandas tqdm
python -c "import nbformat" || pip install nbformat
jupyter notebook membership_probe_FREE.ipynb
```

---

## 📁 Output Files

```
results/
├── probe_results.csv       # All probe scores per text per round
├── auc_results.csv         # AUC per probe per round
├── signal_degradation.png  # 4-panel signal plot
└── auc_degradation.png     # KEY FIGURE — AUC vs paraphrase round
```

---

## 📋 Resume Bullet (add once you have results)

**Membership Inference Probing in Language Models** *(2026)*
- Designed 4 behavioral probes (perplexity, loss curvature, zlib ratio, lowercase ratio)
  to detect training data membership signals in GPT-2 Medium
- Built iterative paraphrase perturbation pipeline using Pegasus to measure
  membership signal degradation across 3 perturbation rounds
- Investigated whether semantic-preserving transformations erase training data signals,
  connecting to active reconstruction methods for LLM data detection
