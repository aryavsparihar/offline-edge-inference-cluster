# Data

## Format

All training and evaluation data is **JSON Lines** (UTF-8, one object per line):

```json
{"text": "मेरी वृद्धावस्था पेंशन पिछले तीन महीने से नहीं आई है", "label": "pension", "lang": "hi"}
```

| field  | type   | meaning |
|--------|--------|---------|
| `text` | string | the user's query, as typed or transcribed (no normalisation required) |
| `label`| string | intent label (see below); any string works, the label set is read from the training file |
| `lang` | string | ISO 639 code: `hi` (Hindi), `ur` (Urdu), `doi` (Dogri) |
| `needs_review` | bool, optional | `true` marks a draft that has not been checked by a native speaker; the compression pipeline skips these |

Blank lines and lines starting with `#` are comments and are ignored by
`edgecluster.data.schema.load_jsonl`.

### Welfare intent labels

| label | covers |
|---|---|
| `pension` | old-age, widow and disability pensions; missing payments |
| `ration_card` | new/duplicate ration cards, adding members, PDS shop issues, One Nation One Ration Card |
| `housing` | PMAY and other housing schemes, flood/displacement repairs, instalments |
| `health_scheme` | Ayushman Bharat / Golden Card, free treatment, maternal health |
| `education_scholarship` | pre-/post-matric scholarships, school admission of displaced children, fee waivers |
| `documents_id` | Aadhaar, birth/domicile/income certificates, voter ID, replacing lost documents |
| `other` | anything else |

## `sample/`: hand-made demo sample (NOT the original data)

`sample/hi.jsonl` (42 Hindi) and `sample/ur.jsonl` (42 Urdu) are **small,
hand-written demo examples** (6 per label) created for this rebuild so the
pipeline, tests and API can run end to end. They are not drawn from the
original Nida-e-Watan corpus, are not real user queries, and are far too
small to train a useful model.

`sample/doi.jsonl` holds 7 **Dogri placeholder drafts** written by a
non-native writer. Each is preceded by `# TODO: native-speaker review` and
carries `"needs_review": true`, so the training pipeline excludes them until a
native speaker has corrected them and removed the flag.

The original regional-language corpus is not included in this repository.

## Public dataset: IndicGLUE `bbca.hi`

To train and evaluate the full pipeline on real data, use AI4Bharat's
IndicGLUE ([`ai4bharat/indic_glue`](https://huggingface.co/datasets/ai4bharat/indic_glue),
Kakwani et al., *Findings of EMNLP 2020*). The Hub serves it as parquet files,
so no loading script is required. It was checked to download and load with
`datasets` before being wired in.

```bash
python -m edgecluster.data.public --config bbca.hi --out data/public
```

This writes `data/public/bbca.hi/{train,validation,test}.jsonl`. `bbca.hi` is
BBC Hindi news articles labeled with 14 topic categories (3,467 train / 866
test). It has no validation split, so 10% of train (stratified, seed 13) is
held out for model selection and pruning-importance scoring. The **test split
is used only for final evaluation**.

Caveats:

* It is a **topic** classification task over news articles, not welfare-intent
  classification over short queries. We use it as a realistic Indic-language
  proxy for measuring how much accuracy compression costs, not as a stand-in
  for the deployed task.
* Articles are much longer than queries, and inputs are truncated to
  `max_seq_len` (128 by default).
* It is Hindi only. Other text configs (`iitp-pr.hi`, `iitp-mr.hi` sentiment,
  `actsa-sc.te`) are also supported via `--config`.

`data/public/` is git-ignored. Check the dataset's license on its Hub page
before redistributing derived files.
