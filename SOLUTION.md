# Reproducibility Instructions
## Environment

The solution was implemented and tested using Python 3.
Required packages:

```bash
pip install numpy scipy
```
### Running the Solution

Make sure the following files are present in the repository:

* `applicant_solution.py`
* `task_and_baseline.py`
* `challenge.mat`

Then run:

```bash
python applicant_solution.py
```

Running the command will:

* load the dataset
* run the baseline method
* run my cancellation method
* evaluate both methods
* generate `results.json`

The final metric used is:

```bash
"baseline": {    3.97, 4.86, 3.48, 3.745 "average_db": 4.017},
  
  "yours":  {
    "per_channel_db": [7.23, 4.76, 9.67, 3.76 "average_db": 6.35
  }
```
The repository is self-contained and does not require modifying the provided task files or dataset.

---
# Final Solution Description

My final solution keeps the provided nonlinear TX cancellation model and adds an additional spatial interference removal stage.

The baseline already models TX-related nonlinear interference reasonably well, so instead of replacing it completely, I focused on improving the remaining residual interference after the baseline prediction.

The process used in the final solution is:

1. Estimate TX-driven interference using the provided helper function
2. Subtract the estimated TX interference
3. Filter the residual into the scoring band
4. Use Singular Value Decomposition (SVD) to estimate the dominant shared interference across RX channels
5. Reconstruct the dominant rank-1 interference component
6. Subtract a controlled portion of that interference from the received signal

The main idea behind the approach is that the external interference described in the task is spatially coherent across channels, so a dominant low-rank structure exists in the residual after TX cancellation.

I used a conservative subtraction factor to avoid removing useful signal content and to remain within the explainability constraints of the evaluator.

The component that contributed most to improving the metric was the rank-1 spatial cancellation stage added after the baseline TX cancellation.

---

# Experiments and Failed Attempts

I tried several other approaches before settling on the final method.

### Rank-2 SVD cancellation

I tested using two spatial modes instead of one.

Although some channels improved significantly, the solution frequently failed the explainability checks and became invalid.

---

### Block-wise / adaptive SVD

I experimented with splitting the signal into smaller sections and estimating different spatial modes for different time regions.

This sometimes improved cancellation locally, but the final output became inconsistent with the evaluator assumptions and often failed validation.

---

### Iterative TX re-estimation

I also tried re-running the TX prediction after partial cancellation.

The idea was to refine the interference estimate after reducing the dominant coherent interference.

While this occasionally improved raw cancellation values, it introduced instability and often reduced explainability below the required threshold.

---

### More aggressive subtraction

Higher subtraction factors produced stronger cancellation in a few channels, especially channel 2, but usually degraded the weaker channels and caused invalid scores.

The final version worked best when the subtraction remained conservative and physically consistent with the challenge assumptions.

---

# Final Notes

One important thing I observed during the challenge is that achieving a higher raw cancellation value is not enough. The removed interference must still remain explainable as a combination of:

* TX-driven nonlinear interference
* spatially coherent interference

The final solution focuses more on stability and validity rather than aggressive cancellation.
