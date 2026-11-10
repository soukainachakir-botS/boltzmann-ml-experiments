# Code for "From Boltzmann to Modern Machine Learning: Exact Connections, Conditions, and Limitations"

This repository contains the code, configurations and raw results of the two experiments in Section IX of the paper.
Every number and table in Section IX can be recomputed from the files in `results/` with

    python analyze.py

## Setup

Python 3.13. Install the exact versions used for the paper:

    pip install -r requirements.txt

All runs used a CPU with two cores (Intel Xeon, 2.1 GHz). No GPU is needed. The scripts set the linear-algebra libraries to one thread, which makes the results reproducible bit for bit on a given machine.

## Data

* Digits: loaded from scikit-learn (`sklearn.datasets.load_digits`), no download needed.
* Fashion-MNIST:

      git clone https://github.com/zalandoresearch/fashion-mnist data/fashion-mnist

* CIFAR-10 (official 50,000/10,000 partition, from a public image mirror):

      git clone https://github.com/YoongiKim/CIFAR-10-images data/CIFAR-10-images
      python data/make_cifar10_official.py

## Experiment A: temperature scaling and calibration

    python experiment_a_calibration.py digits 100 0 digits
    python experiment_a_calibration.py fashion 30 10000 fashion
    python experiment_a_calibration.py cifar10 5 0 cifar10

Configuration (all fixed in the script):

| Item | Digits / Fashion-MNIST | CIFAR-10 |
|---|---|---|
| Splits | 60/20/20 stratified, split seed = run index (0..J-1); Fashion-MNIST: a stratified subset of 10,000 of the 70,000 images per run | official test set (10,000) fixed; validation = stratified 5,000 of the official training set, seed = run index; training = the other 45,000 |
| Preprocessing | standardization with training-set mean and SD | same |
| Logistic regression | L-BFGS, L2, C = 1, max 5,000 iterations | C = 0.01 |
| MLP (NumPy) | 64-32-10 hidden/output layers, ReLU, He-normal init, zero biases, Adam (lr 1e-3), batch 32, 100 epochs, no weight decay, seed = run index | 30 epochs, weight decay 5e-4 |
| Random forest | 200 trees, no depth limit, random_state = run index | same |
| Energies | E = -logit (LR, MLP); E = -log p with vote fractions floored at 1e-4 and renormalized (RF) | same |
| T* | bounded Brent minimization of validation NLL over log T in [log 0.01, log 100] | same |
| Metrics | test accuracy, mean entropy, 15-bin equal-width ECE, NLL; grid T = 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 8, 12, 20, 40 | same |

Confidence intervals and paired tests for the repeated random splits use the corrected resampled t statistic of
Nadeau and Bengio (2003), with n_heldout/n_train = 1/3.

## Experiment B: simulated annealing on Sherrington-Kirkpatrick spin glasses

    python experiment_b_n60_full.py                         # N=60: final energies, trajectories, 16,000-step check, reference energies
    python experiment_b_annealing.py 120 8000 50 40 b120    # N=120
    python experiment_b_annealing.py 200 13333 50 40 b200   # N=200

Couplings J = (A + A^T) / (2 sqrt(N)), A_ij ~ N(0,1), zero diagonal; matrix m uses seed 1000*N + m.
Single-spin Metropolis updates; 66.7 sweeps for every N; schedules (in sweep units, floor 1e-3):
quench T = 0.05; fast 2 * 0.985^t; slow 2 * 0.9985^t; logarithmic 2 / log(2 + t).
Reference energy per matrix: best of 200 anneals of 20,000 steps (T = 2 * 0.9997^t) followed by greedy descent.

## Files

* `experiment_a_calibration.py`, `experiment_b_annealing.py`, `experiment_b_n60_full.py`: experiments
* `data/make_cifar10_official.py`: builds the CIFAR-10 arrays
* `analyze.py`: recomputes all reported statistics (its output is saved in `results/analysis_output.txt`)
* `reliability_rf_digits.py`: the pooled reliability diagram of the random forest (Fig. 12), the pooled ECE, and the check of the floor for zero vote fractions (output in `results/reliability_rf_digits_output.txt`)
* `results/`: raw results (`*_per_split.csv`: one row per run and model; `*_curves.csv`: one row per run, model and temperature;
  `b*_final.npy`: final energies, shape (matrices, schedules, trials); `b60_full.npz`: N=60 results incl. trajectories and reference energies)
