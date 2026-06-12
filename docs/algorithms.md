# ExamGuard v2 — Algorithm Reference

## Detection Architecture: 4-Layer Hybrid

```
Layer 1 (CPU)    Layer 2 (GPU)    Layer 3 (GPU)    Layer 4 (GPU)
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ E1 MinHash│    │ E6 GNN   │    │ XGBoost  │    │ Ollama   │
│ E2 Binom. │    │ E7 VAE   │    │ Ensemble │    │ Narrator │
│ E3 IsoFor.│    │ E8 BERT  │    │          │    │          │
│ E4 IRT 2PL│    │          │    │          │    │          │
│ E5 KDE    │    │          │    │          │    │          │
│ E9 Benford│    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## Engine 1: Copy Ring Detection

**Algorithm:** MinHash LSH → Jaccard + Wrong Answer Agreement → Louvain

**Complexity:** O(N × k) where k = num_perm (128)

### Step 1 — MinHash Locality-Sensitive Hashing
- Problem: N students → N² pairwise comparisons (infeasible for 100K+)
- Solution: MinHash approximates Jaccard similarity in O(N × k)
- Library: `datasketch.MinHash`, `datasketch.MinHashLSH`
- Threshold: 0.7 (candidate filtering)

### Step 2 — Exact Similarity
```
Jaccard:  J(i,j) = |A_i ∩ A_j| / |A_i ∪ A_j|
WAA:      WAA(i,j) = |W_i ∩ W_j| / |W_i ∪ W_j|
Score:    S(i,j) = 0.4 × J + 0.6 × WAA
```

### Step 3 — Louvain Community Detection
- Build graph: edges where Score > 0.75
- Louvain algorithm: maximize modularity Q
- Flag clusters with ≥ 3 members
- Library: `python-louvain` (community)

---

## Engine 2: Statistical Impossibility

**Algorithm:** Binomial Test + Bonferroni Correction

### Herfindahl Index (per-question)
```
p_match_q = Σ(freq_option²)  — probability of random match
p_avg = mean(p_match_q)       — average across all questions
```

### Binomial Test
```
For pair (i,j) with M matching answers out of N questions:
P(X ≥ M) = scipy.stats.binom.sf(M-1, N, p_avg)
```

### Bonferroni Correction
```
α_corrected = 0.05 / C(n_students, 2)
Flag if p_value < α_corrected
```

---

## Engine 3: Center Anomaly Detection

**Algorithm:** Feature Engineering → Isolation Forest + Z-Score

### 8 Features per Center
1. `mean_score` — average student score
2. `score_std` — score standard deviation
3. `top_percentile_rate` — % students in top 10%
4. `pass_rate` — % students above passing threshold
5. `wrong_answer_entropy` — Shannon entropy of wrong answers
6. `high_similarity_pairs` — count of high-similarity pairs
7. `z_vs_national` — z-score compared to national mean
8. `difficulty_correlation` — correlation with expected difficulty curve

### Detection
- `IsolationForest(contamination=0.05, n_estimators=200)`
- Z-score flagging: |z| > 3
- Library: `scikit-learn`

---

## Engine 4: Leak Signature Detection

**Algorithm:** IRT 2-Parameter Logistic Model + Difficulty Gradient

### IRT 2PL Model
```
P(correct) = 1 / (1 + exp(-a(θ - b)))
where:
  θ = student ability (estimated via MLE)
  a = item discrimination
  b = item difficulty
```

### Difficulty Gradient Analysis
```
difficulty_q = 1 - correct_rate_q
Sort into quartiles: Q1 (easy) ... Q4 (hard)
gradient = Q4_accuracy - Q1_accuracy
Flag if gradient > -0.05  (flat = pre-knowledge)
```

### Person-Fit (lz* statistic)
```
lz = Σ[log(P(x_ij | θ)) - E[log(P)]] / sqrt(Var[log(P)])
Flag if |lz*| > 2.0
```

---

## Engine 5: Response Time Analysis

**Algorithm:** KDE + Speed Ratio + K-Means

- KDE: Kernel Density Estimation for per-question time distribution
- Speed ratio: `student_time / median_time`
- Flag: `speed_ratio < 0.2` on hard questions
- K-Means(k=2): separate normal vs pre-knowledge clusters
- Library: `scipy.stats.gaussian_kde`, `scikit-learn`

---

## Engine 6: GNN Fraud Detection

**Algorithm:** 2-Layer GraphSAGE → Node Classification

### Architecture
```
Input: 8-dim node features
  → SAGEConv(8, 64) → ReLU → Dropout(0.3)
  → SAGEConv(64, 32) → ReLU → Dropout(0.3)
  → Linear(32, 2) → Softmax
```

### 8-Dimensional Feature Vector
1. `total_correct / n_questions` — normalized score
2. `score_percentile` — percentile rank
3. `degree` — graph node degree (from k-NN)
4. `avg_similarity` — mean edge weight
5. `center_deviation` — z-score vs center mean
6. `difficulty_gradient` — Q4 - Q1 accuracy
7. `wrong_answer_concentration` — max wrong answer frequency
8. `speed_ratio` — response time metric

### Graph Construction
- k-NN graph: k=10 nearest neighbors by answer similarity
- Edge threshold: similarity > 0.30
- Library: `torch-geometric` (SAGEConv, Data)

---

## Engine 7: VAE Anomaly Detection

**Algorithm:** Variational Autoencoder

### Architecture
```
Encoder: Linear(800,512) → BN → ReLU → Drop(0.3)
       → Linear(512,256) → BN → ReLU → Drop(0.3)
       → Linear(256,128) → BN → ReLU
       → (μ: Linear(128,32), log_σ²: Linear(128,32))

z = μ + σ × ε    (reparameterization trick)

Decoder: mirror of encoder
       → Softmax per question group
```

### Loss
```
L = CrossEntropy_reconstruction + 0.5 × KL_divergence
KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
```

### Anomaly Score
```
anomaly = reconstruction_error per student
threshold = mean + 2.5 × std
```

---

## Engine 8: Question Similarity (NLP)

**Model:** `all-MiniLM-L6-v2` (sentence-transformers)

- Embed question texts using pre-trained BERT
- Compute cosine similarity matrix
- Flag intra-exam duplicates: similarity > 0.85
- Library: `sentence-transformers`

---

## Engine 9: Benford's Law Analysis (Bonus)

**Algorithm:** First-Digit Frequency Test

### Benford's Law
```
P(d) = log₁₀(1 + 1/d)  for d = 1, 2, ..., 9

Expected: {1: 30.1%, 2: 17.6%, 3: 12.5%, 4: 9.7%,
           5: 7.9%, 6: 6.7%, 7: 5.8%, 8: 5.1%, 9: 4.6%}
```

### Chi-Squared Goodness-of-Fit
```
χ² = Σ (observed - expected)² / expected
p-value from chi-squared distribution (df=8)
```

### KL Divergence
```
D_KL(P||Q) = Σ P(x) × log(P(x) / Q(x))
```

### Per-Center Analysis
- Apply Benford test to each center independently
- Flag centers with p-value < 0.01

---

## XGBoost Meta-Ensemble

**12-Dimensional Feature Vector per Student:**

| Feature | Source |
|---------|--------|
| max_similarity | E1 |
| cluster_size | E1 |
| log_pvalue | E2 |
| center_anomaly_score | E3 |
| difficulty_gradient | E4 |
| irt_misfit | E4 |
| speed_ratio | E5 |
| gnn_fraud_prob | E6 |
| vae_anomaly_score | E7 |
| max_q_similarity | E8 |
| num_engines_flagged | All |
| score_percentile | Data |

### Configuration
```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    tree_method='gpu_hist',
    device='cuda',
    max_depth=6,
    n_estimators=200,
    learning_rate=0.1,
    scale_pos_weight=n_clean/n_fraud
)
```

---

## References

1. Benford, F. (1938). "The law of anomalous numbers." *Proceedings of the APS*
2. Nigrini, M. (2012). *Benford's Law: Applications for Forensic Accounting*
3. Hamilton, W. et al. (2017). "Inductive Representation Learning on Large Graphs" (GraphSAGE)
4. Kingma, D.P. & Welling, M. (2013). "Auto-Encoding Variational Bayes" (VAE)
5. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
6. Blondel, V. et al. (2008). "Fast unfolding of communities in large networks" (Louvain)
7. Lord, F.M. (1980). *Applications of Item Response Theory to Practical Testing Problems*
