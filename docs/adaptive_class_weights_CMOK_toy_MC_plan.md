# Adaptive Class-Weight Optimization and a Toy-MC Validation Plan

## 1. Motivation

For classification with finite training samples and finite-capacity models, constant class weights can affect the learned representation even though, in the ideal infinite-data/infinite-capacity limit, they only change class priors and therefore shift logits by class-dependent constants.

This creates an opportunity:

- use class weights as **optimization hyperparameters** to help a finite model allocate capacity;
- analytically remove their prior distortion at inference;
- separately apply the physical event-weight correction, e.g. through a CMOK-style conditional-mean-weight construction.

The central distinction is

\[
\boxed{
\text{optimization class weights}
\neq
\text{physical event weights}.
}
\]

A useful generic training loss is

\[
L_{\rm train}(\theta,\alpha)
=
\sum_{k=1}^{K}
\alpha_k L_k(\theta),
\]

with

\[
L_k(\theta)
=
E_{Q}\!\left[
-\log r_\theta(k|X)
\mid Y=k
\right].
\]

Here:

- \(\theta\): model parameters;
- \(r_\theta(k|x)\): classifier output;
- \(\alpha_k>0\): class-level optimization weights;
- \(Q\): generation distribution.

The \(\alpha_k\) are not physical quantities. Their purpose is to improve the finite-model approximation.

---

# Part I. Strategies for adaptive optimization of class weights

## 2. Baseline fixed class-weight choices

### 2.1 No class weighting

\[
\alpha_k=1.
\]

This lets the generated class proportions determine the relative gradient contribution.

### 2.2 Full class balancing

If the generated class fraction is

\[
\rho_k = Q(Y=k),
\]

then

\[
\alpha_k\propto\frac{1}{\rho_k}.
\]

At the population optimum,

\[
r_k(x)
\propto
\alpha_k q(k|x)
\propto
q(x|k),
\]

so pairwise softmax ratios directly approximate generation-level class-conditional likelihood ratios.

### 2.3 Tempered balancing

A useful continuous family is

\[
\boxed{
\alpha_k\propto\rho_k^{-\beta},
\qquad
0\le\beta\le1.
}
\]

Special cases:

\[
\beta=0
\quad\Rightarrow\quad
\text{no balancing},
\]

\[
\beta=\frac12
\quad\Rightarrow\quad
\text{square-root balancing},
\]

\[
\beta=1
\quad\Rightarrow\quad
\text{full balancing}.
\]

---

## 3. Why class weighting matters in finite models

In the ideal realizable limit,

\[
\operatorname{logit} r(x)
=
\log\frac{p_s(x)}{p_b(x)}
+
\log\frac{\alpha_s}{\alpha_b}.
\]

Thus changing \(\alpha_s/\alpha_b\) only shifts the logit by a constant and leaves the likelihood-ratio ordering unchanged.

For a restricted model \(r_\theta(x)\), however,

\[
\theta^*(\alpha)
=
\arg\min_\theta L_{\rm train}(\theta,\alpha)
\]

depends nontrivially on \(\alpha\). Different class weights alter:

- relative gradient magnitudes;
- representation learning;
- the compromise between regions of phase space;
- sensitivity to statistical fluctuations;
- interaction with regularization;
- interaction with early stopping and SGD noise.

Therefore different class weights can give different finite-sample discrimination performance even though they are asymptotically equivalent after prior correction.

---

## 4. Bilevel optimization of class weights

The cleanest general formulation is

\[
\boxed{
\theta^*(\alpha)
=
\arg\min_\theta
L_{\rm train}(\theta;\alpha)
}
\]

and

\[
\boxed{
\alpha^*
=
\arg\min_\alpha
V_{\rm val}\!\left(\theta^*(\alpha),\alpha\right).
}
\]

The outer validation objective should be computed **after undoing the artificial class prior** and, when relevant, after applying the physical CMOK correction.

Possible outer objectives include

\[
V_{\rm val}
=
\text{physical validation CE},
\]

\[
V_{\rm val}
=
\text{likelihood-ratio calibration error},
\]

\[
V_{\rm val}
=
-Z_A,
\]

or a differentiable proxy for the expected limit.

The conceptual split is

\[
\boxed{
\text{training data optimize }\theta,
\qquad
\text{held-out/meta data optimize }\alpha.
}
\]

The updates may still be interleaved during training.

---

## 5. One-step meta-gradient / gradient-alignment method

Suppose

\[
g_k
=
\nabla_\theta L_k^{\rm train}.
\]

One parameter update is

\[
\theta'
=
\theta
-
\eta_\theta
\sum_k
\alpha_k g_k.
\]

Let

\[
v
=
\nabla_\theta V_{\rm val}.
\]

To first order,

\[
\frac{\partial V_{\rm val}}{\partial \alpha_k}
\simeq
-\eta_\theta\,
v^Tg_k.
\]

Define the class utility

\[
\boxed{
U_k
=
v^Tg_k.
}
\]

Interpretation:

\[
U_k>0
\]

means that increasing the training contribution from class \(k\) tends to move the model in a direction that improves the validation objective.

A practical update is

\[
\log\alpha_k
\leftarrow
\log\alpha_k
+
\eta_\alpha\,U_k.
\]

Using \(\log\alpha_k\) guarantees positivity.

Because a common multiplicative rescaling of all class weights is irrelevant, impose a normalization such as

\[
\frac1K\sum_k\log\alpha_k=0.
\]

---

## 6. CMOK-like conditional-utility construction

Define a noisy per-event training utility

\[
U(X,Y)
=
\nabla_\theta V_{\rm val}^{T}
\nabla_\theta \ell(X,Y;\theta).
\]

If only one constant weight is allowed per class, the relevant object is

\[
\boxed{
\bar U_k
=
E[U\mid Y=k].
}
\]

The first-order effect of class weighting is

\[
V(\theta')
\simeq
V(\theta)
-
\eta_\theta
\sum_k
\rho_k\alpha_k\bar U_k.
\]

This is structurally similar to the CMOK replacement

\[
W
\longrightarrow
E[W|X,Y].
\]

Here the analogous replacement is

\[
U
\longrightarrow
E[U|Y].
\]

| Physical CMOK | Adaptive class weighting |
|---|---|
| noisy physical weight \(W\) | noisy optimization utility \(U\) |
| \(E[W|X,Y]\) | \(E[U|Y]\) |
| physical posterior correction | class-weight update |
| defined by physical measure | defined by validation objective |

The analogy is useful, but the class-weight utility is not uniquely determined by physics. It depends on the outer objective.

---

## 7. Influence-function / full bilevel approximation

Near a trained solution, define

\[
H
=
\nabla_\theta^2
L_{\rm train}.
\]

Then

\[
\frac{d\theta^*}{d\alpha_k}
=
-
H^{-1}
\nabla_\theta L_k,
\]

and

\[
\boxed{
\frac{dV}{d\alpha_k}
=
-
\nabla_\theta V^T
H^{-1}
\nabla_\theta L_k.
}
\]

A per-event influence-style utility is

\[
\boxed{
U_i^{\rm IF}
=
\nabla_\theta V^T
H^{-1}
\nabla_\theta\ell_i.
}
\]

The class-level version is

\[
\boxed{
\bar U_k^{\rm IF}
=
E[
U^{\rm IF}
\mid Y=k
].
}
\]

Interpretation:

- \(\nabla\ell_i\): what direction event \(i\) wants to move parameters;
- \(H^{-1}\): how the finite model responds given its curvature/capacity;
- \(\nabla V\): whether that response helps the final objective.

This is more rigorous but more expensive than the one-step method.

---

## 8. Incorporating finite sample size

For weighted samples, monitor

\[
\boxed{
N_{{\rm eff},k}
=
\frac{
\left(\sum_{i\in k}w_i\right)^2
}{
\sum_{i\in k}w_i^2
}.
}
\]

The uncertainty on the class utility behaves approximately as

\[
\operatorname{SE}(\bar U_k)
\sim
\sqrt{
\frac{\operatorname{Var}(U|k)}
{N_{{\rm eff},k}}
}.
\]

Possible stabilizations:

### 8.1 Shrinkage toward a reference weighting

\[
J_{\rm outer}
=
V_{\rm val}
+
\lambda
\sum_k
\frac{
(\log\alpha_k-\log\alpha_k^{(0)})^2
}{
N_{{\rm eff},k}
}.
\]

### 8.2 Reliability shrinkage

\[
\tilde U_k
=
R_k\bar U_k,
\]

with, for example,

\[
R_k
=
\frac{\tau^2}
{\tau^2+\sigma_k^2/N_{{\rm eff},k}}.
\]

### 8.3 EMA smoothing

\[
\bar U_k^{\rm EMA}
\leftarrow
\lambda_{\rm EMA}\bar U_k^{\rm EMA}
+
(1-\lambda_{\rm EMA})
\bar U_k^{\rm batch}.
\]

---

## 9. Low-dimensional adaptive parameterizations

A simpler family is

\[
\boxed{
\alpha_k
\propto
N_{{\rm eff},k}^{-\beta}.
}
\]

Then only \(\beta\) is optimized.

Another possibility is

\[
\log\alpha_k
=
-\beta\log N_{{\rm eff},k}
+
\gamma I_k^{\rm signal}.
\]

This separates sample-imbalance compensation from explicit signal emphasis.

---

## 10. Removing class weights before CMOK

If the classifier is trained with constant positive class weights \(\alpha_k\),

\[
r_k^{(\alpha)}(x)
=
\frac{
\alpha_k q(k|x)
}{
\sum_j\alpha_j q(j|x)
}.
\]

Recover the generation posterior with

\[
q(k|x)
=
\frac{
r_k^{(\alpha)}(x)/\alpha_k
}{
\sum_j r_j^{(\alpha)}(x)/\alpha_j
}.
\]

In logit form,

\[
\boxed{
z_k^{\rm debiased}(x)
=
z_k^{(\alpha)}(x)
-
\log\alpha_k.
}
\]

If CMOK estimates

\[
\bar w_k(x)=E[W|X=x,Y=k],
\]

then

\[
\boxed{
\pi_k(x)
=
\operatorname{softmax}_k
\left[
z_k^{(\alpha)}(x)
-\log\alpha_k
+\log\bar w_k(x)
\right].
}
\]

Thus class weights can be optimized for training while being removed before physical inference.

---

# Part II. Toy-MC study plan

## 11. Goal

Construct a controlled toy where:

1. the generation distributions are analytically known;
2. the physical distributions differ via event weights;
3. class counts are imbalanced;
4. the exact generation and physical likelihood ratios are known;
5. a deliberately underpowered linear classifier and a moderately flexible NN are tested;
6. four training strategies are compared.

The four requested cases are:

1. plain unweighted CE;
2. class-balanced event-weighted CE;
3. CMOK;
4. CMOK + adaptive class-weight optimization.

---

## 12. Truth-level setup

Use two processes:

\[
Y\in\{S,B\}.
\]

Each event has four observables

\[
x=(x_1,x_2,x_3,x_4).
\]

To make capacity matter, include nonlinear structure.

### 12.1 Signal generation law

Generate

\[
u\sim
\mathcal N(\mu_S,\Sigma_S),
\]

for example

\[
\mu_S=
(0.8,-0.3,0.4,0.0).
\]

Then define

\[
x_1=u_1,
\qquad
x_2=u_2,
\]

\[
x_3=u_3+0.6u_1u_2,
\]

\[
x_4=u_4+0.4u_1^2-0.3u_2^2.
\]

### 12.2 Background generation law

Generate

\[
u\sim
\mathcal N(\mu_B,\Sigma_B),
\]

for example

\[
\mu_B=
(-0.5,0.4,-0.2,0.2),
\]

with a different covariance matrix, and use the same nonlinear map.

This makes the exact LR nonlinear in \(x\), so:

- the linear model is intentionally misspecified;
- the NN should approximate it better.

---

## 13. Generated class imbalance

Use, for example,

\[
N_B^{\rm gen}=1,000,000,
\]

\[
N_S^{\rm gen}=50,000.
\]

Thus

\[
\rho_S\approx0.0476,
\qquad
\rho_B\approx0.9524.
\]

Split each sample into:

- 70% training;
- 15% validation/meta;
- 15% test.

The test set must never be used to optimize adaptive weights.

---

## 14. Physical event weights

Define smooth positive conditional mean weights such as

\[
\bar w_S(x)
=
\exp[
0.35x_1
-0.20x_2
+0.12x_3
],
\]

\[
\bar w_B(x)
=
\exp[
-0.15x_1
+0.30x_2
-0.10x_4
].
\]

Then the unnormalized physical densities are

\[
\tilde p_k(x)
=
q(x|k)\bar w_k(x).
\]

Normalize with

\[
Z_k
=
E_Q[\bar w_k(X)|Y=k],
\]

so

\[
p_k(x)
=
\frac{
q(x|k)\bar w_k(x)
}{
Z_k
}.
\]

The physical LR is

\[
\boxed{
r_{\rm phys}(x)
=
\frac{p_S(x)}{p_B(x)}
=
\frac{q(x|S)}{q(x|B)}
\frac{\bar w_S(x)}{\bar w_B(x)}
\frac{Z_B}{Z_S}.
}
\]

The final constant is irrelevant for ranking.

---

## 15. Noisy per-event MC weights

To make CMOK nontrivial, do not use exactly \(\bar w_k(x)\) as each event's weight.

Use multiplicative log-normal noise:

\[
W
=
\bar w_k(x)
\exp[
\sigma_k\xi-\sigma_k^2/2
],
\qquad
\xi\sim\mathcal N(0,1).
\]

Then

\[
E[W|X,Y=k]
=
\bar w_k(x).
\]

Choose

\[
\sigma_S>\sigma_B
\]

so that the rarer signal class also has poorer effective weighted statistics.

For an optional extension, introduce signed weights with

\[
W
=
\bar w_k(x)
+
\sigma_k\bar w_k(x)\xi
\]

while keeping

\[
E[W|X,Y=k]>0.
\]

Use positive weights for the main study first; add signed weights afterward.

---

# Part III. Models

## 16. Model A: linear classifier

Use logistic regression:

\[
z(x)
=
b+\sum_{j=1}^{4}\theta_jx_j,
\]

\[
r_S(x)
=
\sigma(z(x)).
\]

This cannot represent the exact nonlinear LR.

---

## 17. Model B: moderately complicated NN

Use a small MLP such as

\[
4
\rightarrow
64
\rightarrow
64
\rightarrow
32
\rightarrow
2.
\]

Suggested blocks:

- Linear(4, 64)
- GELU
- LayerNorm
- Linear(64, 64)
- GELU
- Linear(64, 32)
- GELU
- Linear(32, 2)

Optionally add mild weight decay.

The NN should be expressive but not so overpowered that all methods become identical.

---

## 18. CMOK weight head

For Cases 3 and 4, train a separate class-conditioned regression model for

\[
\hat w_k(x)
\approx
E[W|X=x,Y=k].
\]

A simple choice is

\[
4\rightarrow32\rightarrow32\rightarrow1.
\]

For positive weights, use

\[
\hat w_k(x)=\exp[g_k(x)].
\]

Train with MSE or another stable regression loss.

The population optimum of MSE is

\[
\hat w_k^*(x)=E[W|X=x,Y=k].
\]

---

# Part IV. Four experimental cases

## 19. Case 1 — Plain unweighted CE

Train with

\[
L_1
=
-\frac1N
\sum_i
\log r_{y_i}(x_i).
\]

No event weight enters the classifier loss.

Population target:

\[
r_y(x)\rightarrow q(y|x).
\]

For binary classification,

\[
\frac{r_S}{r_B}
=
\frac{\rho_S}{\rho_B}
\frac{q(x|S)}{q(x|B)}.
\]

To compare to the generation LR, remove the generated prior:

\[
\log r_{\rm gen}
=
\log\frac{r_S}{r_B}
-
\log\frac{\rho_S}{\rho_B}.
\]

This should not recover the physical LR when event weights have nontrivial \(x\)-dependence.

---

## 20. Case 2 — Class-balanced event-weighted CE

Use event weights directly in CE and add class balancing:

\[
c_k\propto\frac1{\rho_k}.
\]

Train with

\[
L_2
=
-
\sum_i
c_{y_i}W_i
\log r_{y_i}(x_i).
\]

With positive event weights, the population target is the physical weighted density ratio.

This is the direct weighted-loss baseline.

In the signed-weight extension, this loss may become unstable or ill-posed.

---

## 21. Case 3 — CMOK

Train the classifier with ordinary unweighted CE:

\[
L_{\rm cls}
=
-\sum_i
\log q_\theta(y_i|x_i).
\]

Train a conditional-mean weight estimator:

\[
L_w
=
\sum_i
\left[
\hat w_{y_i}(x_i)-W_i
\right]^2.
\]

At inference construct

\[
\boxed{
\pi_k(x)
=
\frac{
\hat w_k(x)q_\theta(k|x)
}{
\sum_j
\hat w_j(x)q_\theta(j|x)
}.
}
\]

For two classes,

\[
\frac{\pi_S}{\pi_B}
=
\frac{\hat w_S}{\hat w_B}
\frac{q_\theta(S|x)}{q_\theta(B|x)}.
\]

After constant normalization factors, this estimates the physical LR.

---

## 22. Case 4 — CMOK + adaptive class-weight optimization

Train the classifier with

\[
L_{\rm cls}
=
-
\sum_i
\alpha_{y_i}
\log r_\theta(y_i|x_i),
\]

with

\[
\alpha_k=e^{a_k}.
\]

Before CMOK composition, remove the artificial class prior:

\[
z_k^{\rm debiased}
=
z_k^{(\alpha)}
-\log\alpha_k.
\]

Then construct

\[
\boxed{
\pi_k(x)
=
\operatorname{softmax}_k
\left[
z_k^{(\alpha)}
-\log\alpha_k
+\log\hat w_k(x)
\right].
}
\]

---

## 23. Simple adaptive algorithm for Case 4

For the first study, use one free relative parameter

\[
\delta
=
\log\frac{\alpha_S}{\alpha_B}.
\]

Set

\[
a_S=\frac{\delta}{2},
\qquad
a_B=-\frac{\delta}{2}.
\]

### Inner step

For a training minibatch,

\[
L_{\rm train}
=
\alpha_S L_S+\alpha_B L_B.
\]

Compute

\[
\theta'
=
\theta
-
\eta_\theta
\nabla_\theta L_{\rm train}.
\]

### Outer step

On an independent validation/meta minibatch:

1. evaluate at \(\theta'\);
2. debias logits:
   \[
   z_k\to z_k-\log\alpha_k;
   \]
3. apply CMOK:
   \[
   z_k\to
   z_k-\log\alpha_k+\log\hat w_k(x);
   \]
4. compute a physical validation objective.

The simplest outer objective is physical CE:

\[
V_{\rm phys}
=
E_{P,\rm val}
[
-\log\pi_Y(X)
].
\]

Then update

\[
\delta
\leftarrow
\delta
-
\eta_\alpha
\frac{\partial V_{\rm phys}}
{\partial\delta}.
\]

Use a slower learning rate for \(\delta\) than for \(\theta\).

---

## 24. Stabilization for Case 4

Use:

- slow meta learning rate;
- EMA of the hypergradient;
- clipping of \(\delta\);
- optional penalty
  \[
  \lambda_\alpha\delta^2;
  \]
- optional \(N_{\rm eff}\)-based shrinkage.

For the first toy, omit the \(N_{\rm eff}\)-shrinkage term so the basic behavior is easy to interpret.

---

# Part V. Evaluation

## 25. Exact truth benchmarks

Compute exactly:

### Generation LR

\[
r_Q(x)
=
\frac{q(x|S)}{q(x|B)}.
\]

### Physical LR

\[
r_P(x)
=
\frac{p_S(x)}{p_B(x)}
=
r_Q(x)
\frac{\bar w_S(x)}{\bar w_B(x)}
\frac{Z_B}{Z_S}.
\]

All learned scores can therefore be compared directly to truth.

---

## 26. Primary metrics

### 26.1 Physical ROC AUC

Evaluate ROC curves using the physical target distribution.

### 26.2 Correlation with the exact physical log-LR

Compare

\[
\hat\ell(x)
\]

against

\[
\ell_{\rm true}(x)=\log r_P(x).
\]

Use Pearson and Spearman correlations.

### 26.3 MSE in log-LR

After allowing for an irrelevant additive constant,

\[
\operatorname{MSE}_{\log r}
=
E_P[
(\hat\ell-\ell_{\rm true}-c)^2
].
\]

Choose

\[
c
=
E_P[
\hat\ell-\ell_{\rm true}
].
\]

### 26.4 Physical posterior CE / calibration

Compare the predicted physical posterior to the analytically known one.

### 26.5 Binned Asimov significance

Using fixed score bins,

\[
Z_A
=
\sqrt{
2\sum_b
\left[
(s_b+b_b)
\ln\left(1+\frac{s_b}{b_b}\right)
-s_b
\right]
}.
\]

Use identical binning rules for all methods.

---

## 27. Secondary diagnostics

Track:

- train/validation CE;
- class-specific CE;
- gradient norms per class;
- learned \(\alpha_S/\alpha_B\);
- \(N_{{\rm eff},S}\) and \(N_{{\rm eff},B}\);
- CMOK weight-head closure;
- variance across repeated toys.

---

# Part VI. Repeated-toy study

## 28. Why repeated toys are required

One generated dataset cannot distinguish a genuine gain from statistical luck.

Repeat the full experiment for approximately

\[
50\text{--}200
\]

independent toy realizations.

For each model/method record distributions of:

\[
\Delta\operatorname{MSE}_{\log r},
\]

\[
\Delta{\rm AUC},
\]

\[
\Delta Z_A.
\]

Report means, standard errors, and preferably bootstrap intervals.

This is especially important for Case 4 because adaptive hyperparameters can overfit the validation set.

---

# Part VII. Main hypotheses

## 29. Linear model

Because the linear model is misspecified, class weighting should matter strongly.

Expected behavior:

- **Case 1:** approximates the generation posterior under generated imbalance;
- **Case 2:** approximates the physical weighted classification problem directly;
- **Case 3:** recovers the physical law through CMOK, but its classifier representation is learned under the generated imbalance;
- **Case 4:** may improve the finite linear approximation by learning a more useful \(\alpha_S/\alpha_B\).

The largest Case-4 gain is expected here.

---

## 30. Neural-network model

The NN is closer to the realizable regime.

As capacity/statistics increase, properly corrected Cases 2–4 should move toward the same physical likelihood-ratio ordering.

Case 4 may still help under:

- severe class imbalance;
- limited signal statistics;
- early stopping;
- imperfect optimization.

Its gain should generally be smaller than for the linear model if the theory is correct.

---

# Part VIII. Control studies

## 31. Control A — Increase sample size

Scale sample sizes by factors such as

\[
1,\ 5,\ 20.
\]

Finite-statistics differences should shrink as \(N\) grows.

---

## 32. Control B — Increase model capacity

Increase NN width/depth.

If adaptive weighting mainly compensates finite capacity, the Case-4 advantage should decrease.

---

## 33. Control C — Remove \(x\)-dependence of physical weights

Set

\[
\bar w_k(x)=Z_k
\]

constant within each class.

Then generation and physical class-conditional shapes are identical.

CMOK should reduce to a prior/yield correction only.

---

## 34. Control D — Oracle conditional mean

For Cases 3 and 4, replace the learned

\[
\hat w_k(x)
\]

with the exact known

\[
\bar w_k(x).
\]

This separates classifier approximation error from weight-head regression error.

---

## 35. Control E — Signed event weights

After the positive-weight setup is validated, introduce signed individual weights while preserving

\[
E[W|X,Y]>0.
\]

Expected behavior:

- direct weighted CE becomes noisier and may become unstable;
- CMOK remains well-posed because \(W\) is a regression target rather than a loss multiplier.

---

# Part IX. Recommended implementation sequence

## 36. Phase 1 — Validate analytic truth

Implement:

\[
q(x|S),\quad
q(x|B),\quad
\bar w_S(x),\quad
\bar w_B(x).
\]

Numerically check normalization and exact LR calculations.

---

## 37. Phase 2 — Implement both models

Train:

- logistic regression;
- small MLP.

Begin with Case 1 only.

---

## 38. Phase 3 — Add direct weighted CE

Implement Case 2 with positive weights.

Verify that large-statistics, high-capacity behavior approaches the physical LR.

---

## 39. Phase 4 — Add CMOK

Train:

- unweighted classifier;
- conditional-mean weight regressor.

Check closure:

\[
\hat w_k(x)\approx\bar w_k(x).
\]

Then construct the physical posterior.

---

## 40. Phase 5 — Add adaptive class weighting

Use the one-dimensional parameter

\[
\delta=\log(\alpha_S/\alpha_B).
\]

Compare:

- \(\delta=0\);
- fixed class-balanced \(\delta\);
- manually scanned best \(\delta\);
- adaptively learned \(\delta\).

A key sanity check is that the adaptive algorithm should approach the region found by the brute-force scan.

---

## 41. Phase 6 — Repeated toys

Repeat all methods over many independent realizations before interpreting gains.

---

# Part X. Suggested result table

| Model | Method | Physical AUC | log-LR MSE | \(Z_A\) | learned \(\alpha_S/\alpha_B\) |
|---|---|---:|---:|---:|---:|
| Linear | Plain CE | | | | — |
| Linear | Balanced weighted CE | | | | fixed |
| Linear | CMOK | | | | 1 |
| Linear | CMOK + adaptive \(\alpha\) | | | | learned |
| NN | Plain CE | | | | — |
| NN | Balanced weighted CE | | | | fixed |
| NN | CMOK | | | | 1 |
| NN | CMOK + adaptive \(\alpha\) | | | | learned |

---

# Part XI. Suggested plots

1. Learned score versus exact physical log-LR.
2. Residual
   \[
   \hat\ell-\ell_{\rm true}
   \]
   versus exact log-LR.
3. Physical ROC curves.
4. Distribution of learned \(\alpha_S/\alpha_B\) across toy replicas.
5. Validation objective versus a manual scan of
   \[
   \log(\alpha_S/\alpha_B).
   \]
6. Adaptive \(\alpha\) trajectory during training.
7. Performance versus model capacity.
8. Performance versus training sample size.
9. Performance versus
   \[
   N_{{\rm eff},S}/N_{{\rm eff},B}.
   \]

---

# Part XII. Success criteria

The adaptive method is interesting if

\[
\boxed{
\text{Case 4 consistently outperforms Case 3 for finite models}
}
\]

across repeated toys, while its advantage decreases as:

\[
N\to\infty
\]

or model capacity increases.

A particularly convincing pattern would be:

1. clear improvement for the linear model;
2. smaller but measurable improvement for the moderate NN;
3. convergence of Cases 2–4 for sufficiently large samples and sufficiently expressive models.

That would support the interpretation that adaptive class weighting compensates for finite-capacity and finite-statistics imperfections rather than redefining the underlying physical likelihood-ratio target.

---

# Part XIII. Minimal first implementation

For the first prototype, keep the study intentionally simple:

- two classes;
- four variables;
- positive event weights;
- generated imbalance
  \[
  N_B/N_S\sim20;
  \]
- logistic regression;
- MLP
  \[
  4\to64\to64\to32\to2;
  \]
- one adaptive parameter
  \[
  \delta=\log(\alpha_S/\alpha_B);
  \]
- one-step meta-gradient;
- physical validation CE as the outer objective;
- exact analytic physical LR for diagnostics;
- 50 repeated toys.

Only after this works should the study be extended to:

- signed weights;
- many classes;
- influence-function hypergradients;
- \(N_{\rm eff}\)-aware shrinkage;
- direct optimization of significance or expected limits.

---

# 47. Summary

The core strategy is

\[
\boxed{
\text{learn class weights for optimization,
then remove them before physical inference.}
}
\]

In a CMOK-compatible construction,

\[
\boxed{
z_k^{\rm phys}
=
z_k^{(\alpha)}
-\log\alpha_k
+\log\hat{\bar w}_k(x).
}
\]

The class weights can be optimized through a validation-driven bilevel procedure.

A CMOK-like interpretation is to estimate a conditional mean training utility,

\[
\bar U_k
=
E[
\nabla_\theta V^T
\nabla_\theta\ell
\mid Y=k
],
\]

and adjust class weights according to whether emphasizing each class improves the physical validation objective.

The proposed toy study is designed to isolate four effects:

- finite model capacity;
- finite weighted statistics;
- class imbalance;
- mismatch between generation and physical distributions.

The linear model should expose the effect most strongly, while the moderately expressive NN tests whether the benefit persists closer to the realizable regime.
