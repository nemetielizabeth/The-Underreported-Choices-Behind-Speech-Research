## ***The Underreported Choices Behind Speech Research: A Comparison of Linguistic and Acoustic Processing Decisions for Depression Classification in Voice Diaries***

Analysis code for a study of acoustic and linguistic speech features from daily voice diaries with a test case of depression. 

> **❗Status:** Manuscript in preparation. A medRxiv preprint link and full citation will be posted here (expected [August] 2026).

![banner-1](banner-1.png)

## Repository Structure

| Folder | Contents |
|---|---|
| `Preprocessing/` | Audio cleaning, transcript prep, diary selection |
| `Feature-Extraction/` | openSMILE eGeMAPS, WavLM, LIWC-22, RoBERTa embeddings |
| `Classification/` | Nested cross-validated XGBoost pipeline, SHAP, 95% CIs |

## Pipeline

Run in order:

1. **Preprocess**
   `python Preprocessing/preprocess.py`
2. **Extract features**
   `python Feature-Extraction/extract_features.py`
3. **Prepare model input**
  
4. **Classify**
   `python Classification/run_pipeline.py`

## Requirements

Python 3.x. See `requirements.txt`.

## Data

Data are not publicly available due to participant privacy protections. Feature sets will be posted on OSF by the end of August along with the medRxiv preprint. 

## Contact
[enemeti@emory.edy]
