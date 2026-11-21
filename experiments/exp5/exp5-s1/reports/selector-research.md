# Choosing between librcsc WM and the position model: research notes

**Recommendation:** learn the *expected improvement over WM* from observable state, rather than selecting solely from the neural model's uncertainty. Start with an expected-gain regressor and a severity-weighted classifier; compare local-neighborhood methods as transparent baselines. Add a separate tail-harm constraint when severe mistakes matter.

This is a literature-guided proposal, not a claim that one selector has already won the Exp5-S1 experiments. The companion selector study supplies empirical results.

## Why ordinary confidence is not enough

Let the current position errors be:

- e_WM = distance between WM position and the true current position.
- e_NN = distance between model position and the same truth.
- g = e_WM - e_NN. Positive g means that using the model helps.

Ground truth is available **only to form training/evaluation labels**. A live selector sees z: observation ages/masks, last observed motion/body, current unmodified WM state, context, model/WM disagreement, and optionally model uncertainty.

The mean-error-optimal binary choice, absent other costs, is:

**Use NN if E[g | z] > 0; otherwise retain WM.**

This differs from selecting when P(g > 0 | z) is large. For example, a method can win by 1 cm on 90% of cases but lose 1 m on 10%; its mean gain is negative. The loss magnitudes matter. This motivates cost-sensitive deferral [1] and an explicit gain-regression label.

Similarly, high probability that the NN lies within 25 cm of truth does not imply that it improves WM: WM might already be accurate to 1 cm.

### Interpreting the requested confidence thresholds

- Gaussian: a 50/80/90% *coverage level* is chosen for every row; it is not an input-dependent confidence score. A meaningful threshold experiment instead specifies a fixed useful tolerance r (for example 10, 25 or 50 cm), estimates P(distance from the point actually used to truth <= r), and accepts only when that probability exceeds 50/80/90%. Compare empirical reliability as well as the probability model [2].
- Mixture: max component weight >80% says one component dominates the learned mixture. It does **not** mean an 80% chance that the point is within 25 cm. Each component has a spread, components can overlap, and the weighted-mean position can be displaced from the dominant component. Keep the original readout fixed for a gate-only comparison; test a most-probable-component readout as a separate validation-selected candidate, never as an oracle closest-to-truth choice.
- Check every gated result on **all eligible samples**, using WM on rejected rows. Reporting only accepted-row error is not an overall system comparison.

## Ranked methods to try

The ranking is a practical prior for this tabular-feature, two-estimator experiment, not a literature guarantee.

| Priority | Method | What it learns/does | Why useful here | Main limitation |
|---|---|---|---|---|
| 1 | Expected-gain regression | Histogram gradient boosting predicts g from observable features; accept above a validation-chosen threshold | Directly targets pooled position-error improvement; fast non-linear feature interactions [3] | Conditional averages can hide large rare losses |
| 2 | Severity-weighted win classifier | Binary label 1[g>0], sample weight abs(g), then threshold the decision score | Cost-sensitive alternative to ordinary accuracy classification [1] | Weighted probability is **not** ordinary P(NN wins) |
| 3 | Separate error predictors | Learn E[e_WM|z] and E[e_NN|z]; choose the smaller | Inspect which input conditions make each estimator unreliable; easy extension to three NN candidates | Two estimation biases can compound when subtracting |
| 4 | Cluster competence table | Cluster standardized observable features, estimate mean gain and loss frequency in each cluster, default to WM when support is small | Cheap, interpretable regions; approximate local competence [4] | Boundaries are coarse; geometry/age mixes can produce misleading clusters |
| 5 | k-nearest-neighbor gain | On standardized/selected features, average gains of nearby development examples [4,5] | Tests the user's local-similarity idea directly; can reveal local failure pockets | Distance concentration, game dependence and large query cost; do not treat many same-game neighbors as independent evidence |
| 6 | Lower-quantile gain / harm head | Predict a low quantile of g or P(g < -0.25m | z), alongside expected gain | Can trade some mean gain for fewer severe regressions | Predicted quantiles/probabilities need calibration; zero observed failures is not a guarantee |
| 7 | Soft gate between fixed candidates | Learn alpha(z) in [0,1], use p_WM + alpha * (p_NN-p_WM), possibly with a correction-length cap | Allows partial corrections and can preserve accurate WM updates; related to gated experts [6] | A blend may be poor under genuinely separated modes; extra policy needs independent evaluation |
| 8 | Joint selective/defer model with risk calibration | Train predictor and acceptance head together, or calibrate a fixed selector under a defined risk target [7-9] | Later extension if fixed-candidate selectors show remaining headroom | More training complexity; formal risk results require assumptions and cannot be copied to correlated cycles without justification |

**Exact interpretation of severity weighting:** with b=1[g>0] and w=abs(g), weighted-positive mass minus weighted-negative mass equals E[g|z]. Thus the Bayes decision for the weighted classification problem agrees with the mean-gain sign. However, its score is a cost-weighted class probability; calling score 0.8 an ordinary “80% probability NN wins” would be incorrect.

Histogram boosting supports large tabular datasets efficiently and missing values; the official documentation notes its advantages for sample counts above tens of thousands [3]. Cluster/kNN competence adapts the *idea* of dynamic ensemble selection; DESlib itself focuses on classifiers, so do not feed these continuous regression errors into a classification-accuracy API [4].

## Inputs and safeguards

Use only information the observer actually has:

- Last confirmed sighting age and the distinct current WM position/velocity/body counts.
- Validity/missing-field masks, recent observed movement and changes in movement.
- Current unmodified WM position/velocity, model position, and their disagreement magnitude/direction.
- Current named-sighting status and anonymous-refresh risk derived from observable WM/visual information.
- Estimated player/ball/observer geometry and neighbor context; **no true distances**.
- Predicted covariance, confidence radius, or mixture spread/weights, if computed by the frozen models.

A current named visual measurement should remain protected. An anonymous observation should not be assigned to a target using truth. WM position count 0 is observable, whereas the logger's exact target-level anonymous membership may use truth-based matching. Keep those cases separate.

If a neighborhood has little data, or features are far from development data, abstaining to WM is a reasonable policy to test; do not assume it is always superior.

## Reproducible comparison and limits

1. Freeze the candidate NN checkpoints and point readouts.
2. Fit the initial selector on the ten selection games' cached pairs; reserve the other ten validation/calibration games for thresholds/hyperparameters. Do not mix observers of one game across sets. This uses less data than fitting a cross-fitted selector on all 160 training games, but is a practical first experiment.
3. Use the same natural, sample-weighted test population for all policies. Default to WM on rejected rows; rejected rows stay in the denominator.
4. Report mean error, p95/p99, coverage/acceptance, added-error fractions above 0/5/10/25/50/100 cm, maximum added error, and worst-affected situations. Distinguish *total position error* from *extra error compared with WM*.
5. Show an oracle min(e_WM,e_NN) only as an unattainable upper bound. It measures potential headroom; it is not a selector.
6. Compare WM alone, NN alone, simple age/refresh/correction rules, uncertainty-only gates, learned gain selectors and local competence. Validate both the accuracy-oriented and a risk-oriented policy.
7. The original 20 test games have already been examined repeatedly. Their reuse now is an **exploratory diagnostic**. A new game batch or untouched game split is required for a later confirmatory claim.
8. Millions of cycles/observers do not create millions of independent games. Use whole-game checks and, if reporting uncertainty on gains, game-level resampling. No paper below supplies automatic guarantees for these correlated game rows.
9. Conformal Risk Control handles expected monotone losses [8]; Learn then Test frames finite-policy calibration as multiple testing [9]. Their assumptions, loss bounds and calibration/test exchangeability must be checked. Harm *conditional on accepted rows* and population-wide harm are different risks. A selector threshold need not create a monotone empirical harm-rate curve.

Ordinary marginal position-region calibration [2] also does not establish conditional probability of improvement over WM. Calibration can deteriorate under changed teams, policies, visibility settings or physics; empirical work on dataset shift supports that caution, although its experiments were classification tasks [10].

## How to improve training data after the selector study

- Label training cases by signed gain, harm severity and observable situation. Inspect high-loss clusters and near-ball cases before choosing new sampling.
- Add/revisit hard-negative training examples where WM is already good and the model introduces drift; include anonymous/position-only updates and genuine unseen stationary players separately.
- Oversample rare risky training strata or use magnitude-aware loss weights, but preserve natural sample weighting in validation/test and report the changed training objective.
- Collect *new* games targeting underrepresented observable situations (long gaps, difficult near-ball geometry, sharp motion changes) after measuring support. More nearly duplicate easy observations may add little.
- If selecting among models remains hard because observed histories are nearly identical but hidden actions differ, acknowledge irreducible ambiguity; larger networks do not reveal absent action information.
- Use independent game data, or cross-fitted base predictions on training games, when training a more ambitious selector. A gate trained only on optimistic in-sample NN errors may learn an incorrect advantage.
- A lower-risk alternative is a frozen accurate point predictor with an additional error/gain head. Test whether that actually identifies WM superiority rather than only NN's absolute error.

## Verified references

All links below were retrieved successfully during this research pass. Saved HTML snapshots and metadata are in logs/exp5-s1/research/verified-sources.json. Claims above distinguish source findings from our proposed regression adaptation.

1. Mozannar & Sontag (2020), **Consistent Estimators for Learning to Defer to an Expert**. Cost-sensitive learning formulation with a classifier and rejector. [ICML/PMLR](https://proceedings.mlr.press/v119/mozannar20b.html).
2. Kuleshov, Fenner & Ermon (2018), **Accurate Uncertainties for Deep Learning Using Calibrated Regression**. Regression uncertainty calibration; credible intervals need empirical calibration. [ICML/PMLR](https://proceedings.mlr.press/v80/kuleshov18a.html).
3. **scikit-learn Histogram-Based Gradient Boosting**. Official implementation documentation, including regression/quantile objectives and support for missing values. [Documentation](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting).
4. Cruz et al., **DESlib: A Dynamic ensemble selection library in Python** (JMLR 2020). Local competence/dynamic selection implementation context, focused on classification. [Paper](https://arxiv.org/abs/1802.04967).
5. **scikit-learn Nearest Neighbors Regression**. Official implementation context for neighbor-based regression. [Documentation](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression).
6. Jacobs, Jordan, Nowlan & Hinton (1991), **Adaptive Mixtures of Local Experts**. Gated expert combination. [Author-hosted PDF](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), [DOI](https://doi.org/10.1162/neco.1991.3.1.79). The PDF and Crossref metadata were verified separately.
7. Geifman & El-Yaniv (2019), **SelectiveNet: A Deep Neural Network with an Integrated Reject Option**. Joint prediction/rejection, including regression; improved risk–coverage tradeoffs on its benchmark tasks. [ICML/PMLR](https://proceedings.mlr.press/v97/geifman19a.html).
8. Angelopoulos et al., **Conformal Risk Control**. Expected monotone-loss risk control with assumptions. [Paper](https://arxiv.org/abs/2208.02814).
9. Angelopoulos et al., **Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control**. Calibration via multiple testing without base-model refitting. [Paper](https://arxiv.org/abs/2110.01052).
10. Ovadia et al. (2019), **Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift**. Empirical classification study showing post-hoc calibration can fail under shift; relevant warning, not a soccer-regression result. [Paper](https://arxiv.org/abs/1906.02530).
