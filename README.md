


## Observer RMSE DW prediction results

| Method            | Fly Below | Fly Above | Swapping | Swapping (Fast) | Total Avg. |
|------------------|----------|----------|----------|----------------|------------|
| Neural-Swarm    | 1.58     | 1.64     | 3.44     | 2.46           | 2.28       |
| NDP             | **1.27** | 2.18     | 4.01     | 2.06           | 2.38       |
| SO(2)-Equiv.    | 2.37     | 2.14     | 3.36     | 3.14           | 2.75       |
| Empirical       | 2.83     | 5.16     | 7.69     | 7.53           | 5.80       |
| Agile (Proposed) | 1.48     | **1.29** | **2.73** | **1.49**       | **1.75**   |

## Repository Structure

```
📁 arcs
│
├── 📁 controllers
│   └── 📁 nonlinear_feedback
│       ├── 📄 controller_neuralfly.py
│       ├── 📄 controller_neuralswarm.py
│       └── 📄 controller_nfb.py            # used position and px4 attitude controller for final evaluation
│       └── 📄 planner.py                   # generated reference position trajectories for nonlinear feedback controllers
│
├── 📁 data_collection
│   └── 📁 agile_maneuvers
│       ├── 📁 1_flybelow
│       │   └── 📁 speeds_05_20
|       |       ├── 📄 ... .npz             # stored dataset for training
|       |       └── 📄 ...testset.npz       # for validation (not seen during training)
│       ├── 📁 2_flybelow
│       ├── 📁 3_swapping
|       └── 📄 evaluate_models.py
|
├── 📁 observers
│   ├── 📁 SO2
│   │   ├── 📁 SO2_runs             # contains trained models used for evaluation in the report
│   │   ├── 📄 model.py             # core implementation and helper functions in utils.py
│   │   ├── 📄 dataset.py           # to load data
│   |   └── 📄 train_batched.py     # to train observer model once
│   ├── 📁 agile                    # contains proposed observer "AgilePredictor()"
│   ├── 📁 analytical
│   ├── 📁 empirical
│   ├── 📁 ndp
│   ├── 📁 neuralswarm
│   ├── 📁 NNARX                    # experimental
│   ├── 📁 SINDy                    # experimental
│   └── 📄 train_observers.py
│
├── 📁 other                # old scripts that could still be interesting
├── 📁 uav
│   └── 📄 uav.py
├── 📁 utils
│   └── 📄 utils.py
│
└── 📄 README.md
```
### Observers

- In `arcs/observers/` are reimplemented observers for the downwash disturbance, and some experimental (SINDy and NNARX, that did not make it to the final report but showed promising results).

- Each observer contains a `model.py`, `dataset.py`, and a `train.py` or `train_batched.py` scripts to load and train on a dataset.

- `arcs/observers/train_observers.py` can be run to train one target observer for different seeds and hyperparameters.

### Dataset

- In `arcs/data_collection/agile_maneuvers/` contains 3 scenarios `flybelow`, `flyabove`, `swapping`.

- All contain data collection flight maneuver script, e.g.`1_flybelow.py` to collect DW data.

- For each scenario, dataset is stored as ".npz" in e.g. "1_flybelow/speeds_05_20/", where "...testset.npz" is the validation dataset.

- `3_swapping` has also `speeds_05_20` and `speeds_20_40` that have data for flight 2-4 m/s flight trajectories.

- See `uav.py` and e.g. `observers/ndp/dataset.py` files for how to load UAV states and external forces from stored file as example.

### UAV

- In `arcs/uav/uav.py` is the UAV class for reference about the UAV state structure.

### Utils

- In `arcs/utilts/utils.py` contains all math, encoding, plotting and dataset loading functions for the scripts above.
