# Corrected Report Notes

These outputs follow the report's main idea, but recalculate the values from the current CSV rather than trusting the numbers already written in the Word document.

## Classification
- Full Greens model ROC-AUC: 0.873 +/- 0.043.
- Full Greens model balanced accuracy: 0.848 +/- 0.034.
- Sensitivity model without Q8, Q20, Q25_4 and Q26_4 ROC-AUC: 0.721 +/- 0.065.
- Sensitivity model without those direct party/warmth signals balanced accuracy: 0.668 +/- 0.054.
- Improvement to the report: keep the full tree as a pattern-discovery model, but explicitly say that much of its strength comes from party identity, previous vote and direct Greens/Adam Bandt warmth. The restricted model is the better evidence for non-obvious attitudinal signal.

## Association Rules
- Rules are recalculated from transactions built from the survey columns. Thermometers are binned into 0-2, 3-4, 5, 6-7 and 8-10; Q13 and Q39 are binned into three ordered groups.
- Improvement to the report: report both direct-signal rules and restricted rules. The direct-signal rules are interpretable but close to tautological.

## Clustering
- Best k by silhouette in the recalculated run: k=2.
- Silhouette for k=2: 0.120; silhouette for k=3: 0.105.
- Improvement to the report: if k=3 is used for interpretability, describe it as a descriptive segmentation rather than as evidence of sharply separated natural groups. If the marking focus is formal quality, prefer the best-silhouette solution.

## General Caveats
- The analysis is unweighted; W1 is not used in model fitting.
- Accuracy is not the best headline metric because Greens voters are a minority class.
- Do not present the model as causal or suitable for individual-level political targeting.
