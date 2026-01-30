# Audio-Visual Affect Modelling (using the RAVDESS dataset)

## Overview

This project implements a baseline pipeline for facial-expression-based emotion classification from video, with a focus on evaluation methodology and generalization across individuals.

Using the RAVDESS dataset, facial landmarks are extracted from sampled video frames and summarized into a compact feature representation. A simple classifier is then trained to predict emotion labels under two evaluation regimes:
	1.	A standard random train/test split
	2.	An actor-wise split where identities do not overlap between training and test sets

This allows direct investigation of within-person versus between-person generalization and the extent to which apparent performance may rely on identity-specific cues rather than emotion-related dynamics.

## Landmark Extraction Example

Below is an example frame with detected facial landmarks overlaid. Each landmark corresponds to a 3D point used as input features for the classifier after temporal aggregation.

![Landmark example](assets/frame_07_idx_087_points.jpg)

## Key results

Feature set | Random split | Actor-wise split | Drop
------------|--------------|------------------|------
All features (mean + std) | 0.819 | 0.600 | 0.219
Mean-only features | 0.795 | 0.527 | 0.269
Std-only features | 0.552 | 0.410 | 0.142

## Interpretation

Performance is substantially higher under random splitting than under actor-wise evaluation for all feature representations, indicating that identity-specific information contributes strongly to apparent predictive accuracy.

Mean landmark coordinates yield relatively high random-split accuracy but show the largest drop under actor-wise evaluation, suggesting that static facial geometry contains substantial identity-related signal that does not generalize well to unseen individuals.

Using only landmark variability reduces overall accuracy but also reduces the generalization gap, indicating that dynamic movement patterns are less tied to individual identity but also less informative on their own for emotion classification in this dataset.

Together, these results illustrate that naive evaluation strategies can substantially overestimate generalization performance and that disentangling person-specific baselines from expression-related dynamics is central for robust affect modelling.

## Method Overview

1. Frame Sampling

Ten evenly spaced frames are sampled from each video clip.

2. Facial Landmark Extraction

MediaPipe Face Landmarker is used to extract 468 three-dimensional facial landmarks per frame.

3. Temporal Aggregation

For each landmark coordinate, the mean and standard deviation across frames are computed, yielding a fixed-length feature vector per video.

4. Classification

A logistic regression classifier is trained to predict one of eight emotion categories.

5. Evaluation

Performance is compared under:
	•	random train/test split
	•	actor-wise (identity-disjoint) split

Feature ablations are performed using:
	•	mean-only features
	•	standard-deviation-only features

## Repo structure

src/
  extract_face_landmarks_tasks.py   # video → landmark features
  train_eval_landmarks.py           # classifier + evaluation

data/
  raw/        # expected locally (not versioned)
  processed/  # generated features (not versioned)

models/       # downloaded MediaPipe model (not versioned)

## Reproducibility

To reproduce the results:
python src/extract_face_landmarks_tasks.py
python src/train_eval_landmarks.py

(The RAVDESS dataset must be downloaded separately and placed in data/raw/.)

## Notes and Limitations
	•	RAVDESS contains acted emotional speech recorded under controlled conditions; results may not transfer directly to naturalistic settings.
	•	The goal of this project is not state-of-the-art emotion recognition performance but to demonstrate how evaluation strategy critically affects conclusions.
	•	The pipeline serves as a baseline for future extensions such as temporal modelling, multimodal fusion, or within-person change analysis.

## Motivation

This project was designed as a methodological exercise aligned with research on dynamic wellbeing and affect modelling. It demonstrates concretely how naive evaluation can inflate performance estimates and obscure the distinction between individual-specific patterns and general emotional dynamics.
