# Automated Squash Referee Assistant

### Classifying squash rallies as stroke, yes-let or no-let with a CNN + GRU/LSTM video model

## Project Overview

| Item | Details |
|---|---|
| **Project name** | Automated Squash Referee Assistant |
| **Team name** | GameVision AI |
| **Team members** | Sahitya Sharma · 2024A7PS0096H<br>Arnav Gupta · 2024A8PS0487H<br>Aarav Mathur · 2024A3PS0336H<br>Goru Nikhil · 2024A3PS0357H<br>Jay Kimtani · 2024A3PS0392H<br>Priyansh Thadani · 2024A3PS0356H<br>Khushal Kadyan · 2024ADPS0181H |
| **Declared topic area** | Applications Of Deep Learning |
| **Track** | Application |
| **Dataset** | Squash Fouls Dataset (Kaggle), `black_and_white` folder |
| **GitHub repository** | Add repository link here |

## Short Description

In squash, players often get in each other's way, and the referee has to decide on the spot whether a rally continues (**stroke**), is replayed (**yes-let**) or whether the appeal is rejected (**no-let**). These calls are subjective and a common source of arguments.

We will train a model that watches a short rally clip and predicts one of those three outcomes. A pretrained CNN turns each frame into a feature vector, and a GRU or LSTM reads those vectors in order, so the model can pick up on how the players move across the whole clip.

The idea is to use it as a **referee assistant** that flags contentious calls and offers a second opinion, not to replace the referee.

---

## 1. Problem Statement and Motivation

Squash is played by two people sharing one small court, so they cross each other's paths constantly. Quite often one player says they could not reach or swing at the ball because the opponent was in the way. The referee then has to pick one of three outcomes within a second or two.

- If the interference cost the player a winning shot, a **stroke** is awarded and the point goes to them.
- If the player could have played the ball but was blocked, the rally is replayed, which is a **yes-let**.
- If the appeal is not justified, play stands as it was, a **no-let**.

These decisions are hard because they depend on small details: where each body is, how the swing was going, and whether the player really made an effort to get to the ball. Players and fans argue about them frequently, and in professional matches a single call can change the result. Smaller clubs and tournaments usually have no video review, so the on-court referee has one look and no replay.

We want to find out whether a model can learn to make this call from video alone. Given a short clip, it outputs **stroke**, **yes-let** or **no-let**.

We see it as a helper for the referee. It could flag clips where its prediction disagrees with the on-court call, or where it is not confident, and give an AI-based second opinion.

The problem fits the course because the answer is not visible in any single frame. It only shows up in how movement unfolds over time. A model therefore has to extract spatial features from each frame with a CNN and then reason over the sequence with a recurrent network.

---

## 2. Course Concepts and Techniques

### CNN with Transfer Learning

A **ResNet-50** pretrained on ImageNet [3, 4] acts as a frozen feature extractor. If time allows, we will also fine-tune its last block.

### GRU and LSTM Networks

A **Gated Recurrent Unit (GRU)** [2] and a **Long Short-Term Memory (LSTM)** network [1] read the frame features in order. Their gates let gradients survive over longer sequences, which is why we prefer them to a plain RNN. They are trained with backpropagation through time.

### Supervised Multi-Class Classification

A softmax output layer with cross-entropy loss, optimised with **Adam**.

### Regularisation

Because the dataset is small, we use:

- Dropout
- Weight decay
- Early stopping on a validation set
- Light data augmentation

### Model Evaluation

Evaluation will include:

- Train/validation/test splits
- Confusion matrix
- Precision
- Recall
- Macro F1
- Ablation studies

---

## 3. Proposed Approach and Methodology

### 3.1 Data

We use the `black_and_white` folder of the **Squash Fouls Dataset** on Kaggle [9]. It holds short squash clips sorted into the three classes.

Before modelling, we will audit:

- Number of clips per decision
- Clip length
- Frame rate
- Number of different matches
- Number of camera views

The clips are grayscale, so each frame will be copied across three channels to fit the pretrained CNN.

If the classes are imbalanced, we will use a **class-weighted loss** and report macro-averaged metrics instead of plain accuracy.

### 3.2 Pipeline

#### 1. Frame Sampling

Take a fixed number of evenly spaced frames per clip, for example **16 or 32 frames**.

- Resize to `224 × 224`
- Normalise using ImageNet statistics
- Pad and mask shorter clips

#### 2. Feature Extraction

Run every frame through the frozen **ResNet-50** and keep the **2048-dimensional pooled vector**.

These features are computed once and cached so that training the sequence model is quick and fits on a free Colab or Kaggle GPU.

#### 3. Temporal Model

The feature sequence goes into a **1- or 2-layer GRU** and, separately, an **LSTM** with a hidden size of about **256**.

The sequence model is followed by:

- Dropout
- A fully connected layer
- 3 output classes

We will also try bidirectional versions.

#### 4. Referee-Assistant Layer

A clip is flagged as contentious when:

- The top softmax probability is below a threshold, or
- The top two class probabilities are close

The threshold will be selected using the validation set.

This allows the system to focus the referee's attention on uncertain or potentially contentious clips.

### 3.3 Tools and Environment

**Language and libraries**

- Python
- PyTorch
- torchvision
- OpenCV
- scikit-learn
- Matplotlib

**Compute**

Free **Google Colab** or **Kaggle GPUs** are sufficient. Because CNN features are cached, only the smaller GRU/LSTM is trained repeatedly, and one run should take minutes rather than hours.

**Collaboration**

All code will live in the team GitHub repository.

The team will:

- Work on branches
- Merge through pull requests
- Keep fixed random seeds
- Keep split files in the repository

This makes results reproducible.

### 3.4 Evaluation

The data will be split into approximately:

- **70% train**
- **15% validation**
- **15% test**

The split will be stratified by class.

To avoid leakage, clips from the same match or rally will stay in the same split wherever the dataset lets us identify them.

We will report:

- Accuracy
- Per-class precision
- Per-class recall
- Macro F1
- Confusion matrix

Special attention will be given to **yes-let vs no-let**, which we expect to be the hardest pair.

#### Baselines and Ablations

**Baseline A:** Average the frame features over time and classify with logistic regression or a small MLP, with no temporal modelling.

**GRU vs LSTM:** Compare:

- Unidirectional vs bidirectional
- 8, 16 or 32 sampled frames

**Frozen vs partly fine-tuned CNN:** Compare with and without:

- Flip
- Small crop
- Brightness augmentation

We will count the project as a success if the best GRU/LSTM model clearly beats Baseline A on macro F1 on the test set, and if confidence-based flagging catches more of the model's mistakes than flagging clips at random.

Misclassified clips will also be reviewed manually to understand where and why the model fails.

---

## 4. Expected Results

We do not want to promise numbers before seeing the data. The following are the expected trends, and actual figures will be reported after the baseline is trained.

| Experiment | Expected result | Reasoning |
|---|---|---|
| **Baseline A (mean-pooled features)** | Clearly above chance, but weak on yes-let vs no-let | Averaging throws away event order, so it can see what appears in the clip but not how it unfolds. |
| **GRU / LSTM** | A noticeable gain in macro F1 over the baseline | The label depends on movement over time, which recurrent models are built to capture. |
| **GRU vs LSTM** | Similar scores, with GRU a bit faster to train | Clips are short and the dataset is small, so the extra LSTM capacity may not help. |
| **More frames (8, 16, 32)** | Gains that level off after about 16 | More frames add detail, but also noise and a higher risk of overfitting. |
| **Fine-tuning the CNN** | Small gain, with possible overfitting | Pretrained features are general, and the grayscale dataset is small. |
| **Confidence-based flagging** | Flagged clips have a higher error rate than unflagged ones | Ambiguous clips should produce less confident predictions. |

Realistically, we expect the model to be good but not perfect. Let decisions are subjective even for human referees, so some disagreement with the dataset labels is unavoidable.

A model that is right most of the time and honest about the clips it is unsure of can still be useful as a second opinion.

### Main Risks

- The dataset is small.
- The labels are ambiguous.
- Grayscale footage may limit what an ImageNet model has learned to look at.

The project addresses these using cached features, regularisation, class weighting and a baseline-first workflow so that problems show up early.

### 4.1 Risks and Mitigation

| Risk | Why it matters | What we will do |
|---|---|---|
| **Small dataset** | The model may memorise clips instead of learning movement patterns | Frozen CNN, dropout, weight decay, early stopping, light augmentation |
| **Class imbalance** | The model could favour the majority class and still score a high accuracy | Class-weighted loss, macro F1 and per-class recall |
| **Data leakage** | Clips from the same rally in train and test would inflate results | Split by match or rally where possible and document how |
| **Ambiguous labels** | Even human referees disagree on some let calls | Report errors openly and review disputed clips by hand |
| **Falling behind schedule** | Seven weeks is short for a full deep learning pipeline | Baseline first, cached features, weekly check-ins, simple fallback scope |

### Fallback Plan

If fine-tuning or bidirectional models do not fit in the available time, the project will still deliver the core result:

> A CNN + GRU/LSTM classifier compared against a no-temporal baseline.

---

## 5. Current Work in the Area

Combining a CNN with a recurrent network for video is a well-established idea.

- Karpathy et al. [6] studied CNNs for large-scale video classification.
- Donahue et al. [5] introduced the Long-Term Recurrent Convolutional Network, in which per-frame CNN features feed an LSTM.
- Ng et al. [7] showed that LSTMs over frame features can use long clips better than simply pooling short snippets.
- More recent models such as **I3D** [8] learn motion directly with 3D convolutions. They are more powerful, but they need substantially more data and compute than this project has available.

Our CNN + GRU/LSTM design follows the lighter approach in [5, 7], which suits a small dataset and a student budget.

The **Squash Fouls Dataset** [9] was published on Kaggle for computer vision and machine learning work. We did not find a published paper that uses it specifically for let decisions, so the decisions themselves follow the **World Squash Federation rules** [10].

---

## 6. Feasibility: Timeline and Roles

The team has seven members split into four groups, each owning one part of the project. Groups work in parallel where possible, and the full team meets once a week to check progress against milestones.

### 6.1 Roles

| Group | Members | Responsibilities | Main output |
|---|---|---|---|
| **A. Dataset review and preprocessing** | Arnav Gupta, Jay Kimtani | Audit classes and clip lengths, build leakage-safe splits, sample frames, resize and normalise, set up augmentation | Data report, split files, preprocessing script |
| **B. Model making** | Sahitya Sharma, Goru Nikhil | Extract and cache ResNet-50 features, build GRU and LSTM models, write the training loop | Cached features, model code, checkpoints |
| **C. Model testing** | Aarav Mathur, Priyansh Thadani | Build Baseline A, run ablations, compute metrics and confusion matrices, analyse errors, tune flagging threshold | Results tables, plots, error analysis |
| **D. Report writing** | Khushal Kadyan | Maintain the README and repository, write the final report and slides, keep track of references and milestones | Final report, slides, README |

### 6.2 Week-by-Week Milestones

| Week | Milestone | Lead group | Deliverable |
|---|---|---|---|
| **1** | Set up repository and environment, download data, audit classes, clip lengths and cameras | A (D helps) | Repo, data report |
| **2** | Build leakage-safe splits, frame sampling and augmentation pipeline | A | Split files, preprocessing script |
| **3** | Extract and cache CNN features; train Baseline A | B and C | Cached features, baseline results |
| **4** | Build and train first GRU and LSTM models end to end | B | First checkpoints |
| **5** | Tuning and ablations: frames, bidirectional, fine-tuning, augmentation | B and C | Ablation table |
| **6** | Final test evaluation, error analysis, referee-assistant flagging demo | C | Test results, demo |
| **7** | Final report, slides, code cleanup and README | D (all help) | Final submission |

---

## 7. Dataset

**Squash Fouls Dataset:**  
https://www.kaggle.com/datasets/youssefmohamed55555/squash-fouls-dataset

The project uses the `black_and_white` folder containing short clips organised into the three target classes:

- Stroke
- Yes-let
- No-let

---

## 8. Project Pipeline

```text
Squash Rally Video
        │
        ▼
  Frame Sampling
  (8 / 16 / 32 frames)
        │
        ▼
  Resize + Normalise
      224 × 224
        │
        ▼
 Grayscale → 3 Channels
        │
        ▼
 Frozen ResNet-50
 2048-D/frame feature
        │
        ▼
 Cached Feature Sequence
        │
        ├───────────────┐
        ▼               ▼
      GRU              LSTM
        │               │
        └───────┬───────┘
                ▼
           Dropout + FC
                │
                ▼
         3-Class Softmax
                │
                ▼
      Stroke / Yes-let / No-let
                │
                ▼
   Confidence / Ambiguity Flag
                │
                ▼
       Referee Second Opinion
```

---

## 9. References

1. S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

2. K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” *EMNLP*, 2014.  
   https://arxiv.org/abs/1406.1078

3. K. He, X. Zhang, S. Ren and J. Sun, “Deep residual learning for image recognition,” *CVPR*, 2016.  
   https://arxiv.org/abs/1512.03385

4. J. Deng et al., “ImageNet: A large-scale hierarchical image database,” *CVPR*, 2009.

5. J. Donahue et al., “Long-term recurrent convolutional networks for visual recognition and description,” *CVPR*, 2015.  
   https://arxiv.org/abs/1411.4389

6. A. Karpathy et al., “Large-scale video classification with convolutional neural networks,” *CVPR*, 2014, pp. 1725–1732.

7. J. Yue-Hei Ng et al., “Beyond short snippets: Deep networks for video classification,” *CVPR*, 2015.  
   https://arxiv.org/abs/1503.08909

8. J. Carreira and A. Zisserman, “Quo vadis, action recognition? A new model and the Kinetics dataset,” *CVPR*, 2017.  
   https://arxiv.org/abs/1705.07750

9. Y. Mohamed, “Squash Fouls Dataset,” Kaggle.  
   https://www.kaggle.com/datasets/youssefmohamed55555/squash-fouls-dataset

10. World Squash Federation, “Rules of Squash.”  
    https://www.worldsquash.org
