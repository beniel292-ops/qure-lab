# Demo script (about 3 minutes, desktop)

**Before the demo:**

- Open the published dashboard link, or `web/qure_lab_standalone.html` (works offline, double-click).
- Keep `results/final/figures/fig_test_noise_mitigation.png` ready as a static fallback.

**Steps:**

1. **Header.** Read the research question. Point at the status chips: real data, simulated execution, hypothetical noise, test split.
2. **Case explorer, the default case** (it is flagged at N4).
    - Show the reader outlines. Say: "No CT pixels; these are the radiologists' contours."
    - Show the four features turning into angles.
3. **Execution results.**
    - Click N0: the noiseless reference sits just below the threshold.
    - Click N4: the noisy runs (red) cross the threshold and the mitigated runs (blue) come back. Point out that this case's noiseless answer **disagrees with the label**: mitigation restores the model's answer, not the truth.
4. **Filters.**
    - Choose "Mitigation moved score farther away" at N2. Mitigation doesn't help with gate noise.
    - Choose "Noiseless model wrong vs label" to show that unfavourable cases are visible.
5. **State diagnostics.** Open the panel. Say: "These describe qubit states, not confidence."
6. **Aggregate results.**
    - The accuracy chart: the quantum model is level with logistic regression.
    - The cost table: mitigation lowers deviation and flips but doubles the spread.
    - The instability rule table.
7. **Method and limits.** Close on the limitations list.

**If anything fails:** use the standalone HTML file (no network), then the PNG figure.
