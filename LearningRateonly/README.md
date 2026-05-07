# Differentially Private Learning Rate Tuning

This part computes the noise scale (**σ, sigma**) required to achieve a desired privacy level (**ε, epsilon**) using a forward-sweep and interpolation approach.

---
## First Step: Inverse Privacy Parameter Estimation
###  Script

#### `run_inverse.py`

For a fixed set of target epsilon values, the script finds the corresponding sigma values for a chosen method (variant).

---

###  Usage

Run the script from the command line by specifying the variant:

```bash
python run_inverse.py --variant variant1
python run_inverse.py --variant variant2
python run_inverse.py --variant baseline
python run_inverse.py --variant single_run
```
## Second Step: Tuning and Training Models For Each Setup
This part runs poisson sampled tuning runs on the subsampled dataset and saves their weights.

```bash
python tuning_script_all.py
```
Continues training on the whole dataset according to the mechanism of Variant2 and uses the pretrained weights from hyperparameter tuning.
```bash
python opacus_mnist_avgruns.py
```

and optionally for plotting 
```bash
python plot_results_avgruns.py
