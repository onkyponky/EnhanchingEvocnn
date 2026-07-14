# EvoCNN + NSGA-II + SAMR

**Automated CNN design through multi-objective evolutionary search.**

This project extends [EvoCNN](https://doi.org/10.1109/TEVC.2018.2869189) (Sun et al., 2019) with two additions: **NSGA-II** for multi-objective selection, and **Self-Adaptive Mutation Rates (SAMR)** so the search tunes its own mutation behaviour. Instead of returning one "best" CNN, it returns a Pareto front — a spread of architectures trading accuracy against parameter count, so you can pick the one that fits your hardware budget.

The original EvoCNN codebase was also fully migrated from TensorFlow 1.x / Python 2.x to **TensorFlow 2.19 / Python 3.12**, which is what makes GPU training and eager execution possible here.

> Master of Computer Science research project, University of Adelaide (Semester 2, 2025).
> Author: Vedansh Kumar · Supervisor: Indu Bala

---

## Why

Designing a CNN by hand is slow, and accuracy is not the only thing that matters. A model that hits 93% but needs 1.7M parameters is useless on a microcontroller. EvoCNN optimises accuracy alone, with a fixed mutation rate — so it can't express that trade-off, and it can't adapt its own search behaviour. This project fixes both.

**Two objectives, minimised jointly:**

```
f1(i) = -validation_accuracy(i)
f2(i) =  trainable_parameters(i)
```

---

## Results (Fashion-MNIST)

| Model | Parameters | Val. Accuracy |
|---|---|---|
| LeNet (baseline) | 455 K | 91.5% |
| 3-layer CNN (baseline) | 1.7 M | 93.6% |
| **Evolved (Pareto set)** | **4.8 K – 27 K** | **85 – 89%** |

The best evolved individual reached **88.7% accuracy at ~27 K parameters**; the smallest viable network hit **85.8% at ~4.8 K parameters** — roughly **350× fewer parameters** than the 3-layer baseline for about 5 points of accuracy. For edge deployment that's a trade most people would take.

Mean mutation rate fell from ~0.21 in early generations to ~0.09 by generation 5, without anyone tuning it — broad exploration first, refinement later, which is exactly what SAMR is supposed to do.

---

## How it works

```
Data prep  →  Baseline training  →  Evolutionary loop  →  Pareto analytics
                                    ├─ Evaluate  (3 epochs per individual)
                                    ├─ Vary      (SAMR crossover + mutation)
                                    └─ Select    (NSGA-II)
```

**Encoding.** Each individual is a variable-length chromosome: conv / pool / dense genes with their hyperparameters, plus one real-valued mutation-rate gene `r`, initialised uniformly in `[0.05, 0.30]`.

**SAMR.** Before every mutation call, `r` self-adapts with a log-normal update:

```
r ← clip( r · exp(τ · N(0,1)),  5e-4,  0.7 ),   τ = 0.1
```

`r` then controls both the *probability* of structural mutation (add / modify / delete a layer) and the *magnitude* of parametric perturbation (filters, kernel, stride). After crossover, offspring inherit the mean of both parents' rates, re-perturbed. No manual mutation tuning anywhere in the pipeline.

**NSGA-II.** Parents and offspring are merged into a pool of 2N, ranked by fast non-dominated sorting, then split by crowding distance. Boundary solutions get infinite crowding distance, so the extremes of the front (smallest model, most accurate model) always survive. Parents for crossover are picked by binary tournament on rank + crowding distance.

---

## Setup

```bash
git clone <repo-url>
cd evocnn-nsga2-samr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.12.3 · TensorFlow 2.19 · NumPy 2.3.3 · Matplotlib 3.10.7

Developed and tested on an NVIDIA RTX 4050 Laptop GPU (3.5 GB VRAM) under WSL2 with CUDA. It will run on CPU, just slowly.

---

## Usage

```bash
# 1. Train the fixed baselines (LeNet + 3-layer CNN)
python train_baselines.py

# 2. Run the evolutionary search
python main.py

# 3. Generate Pareto fronts and trend plots
python plot_results.py
```

### Output

```
save_data/
├── baseline_metrics.csv          # baseline accuracy / params
├── generation_summary.csv        # best & mean accuracy, complexity, mutation rate per gen
└── gen_000/ … gen_005/
    ├── pop.txt                   # human-readable population log
    ├── pop.dat                   # pickled population (full reproducibility)
    └── indi_k_best_model.keras   # checkpointed best individual
```

---

## Configuration

| Setting | Value |
|---|---|
| Dataset | Fashion-MNIST (60 K train / 10 K val) |
| Population size | 10 |
| Generations | 5–10 |
| Epochs per individual | 3 (partial-training proxy) |
| Optimiser | Adam, lr = 1e-3 |
| Batch size | 128 |
| Loss | SparseCategoricalCrossentropy(from_logits=True) |
| Mutation rate | self-adaptive, τ = 0.1, clipped to [5e-4, 0.7] |

---

## Known limitations

Being upfront about these, since they matter if you want to build on the work:

- **Validation/test overlap in the evolutionary runs.** Evolution uses the 10 K test split as its validation set, while the baselines use a separate 55 K / 5 K / 10 K split. The two are therefore not measured on identical data, and evolved accuracies carry a mild optimistic bias.
- **3-epoch training proxy.** Standard practice in NAS, but it ranks architectures on partial convergence, not final performance.
- **Seeding is not globally enforced** in `main.py`, so evolutionary runs are not bit-for-bit reproducible. The baselines are seeded.
- **Fashion-MNIST only.** Whether these gains hold on CIFAR-10 or Tiny-ImageNet is untested.
- **Two objectives only.** Latency, FLOPs, and energy — the things that actually decide edge deployment — are not yet optimised.

## Next steps

Scale to CIFAR-10 / Tiny-ImageNet · add latency and FLOPs as objectives for true hardware-aware search · layer-wise mutation rates · asynchronous / island-model evolution for wall-clock speedup.

---

## References

- Sun et al. (2019). *Evolving Deep Convolutional Neural Networks for Image Classification.* IEEE TEVC 23(4).
- Deb et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.* IEEE TEVC 6(2).
- Doerr & Doerr (2021). *Runtime Analysis for Self-Adaptive Mutation Rates.* Evolutionary Computation.
- Carles-Bou & Galán (2023). *Self-Adaptive Polynomial Mutation in NSGA-II.* Soft Computing 27(23).
- Xiao et al. (2017). *Fashion-MNIST.* Zalando Research.

## License

MIT
