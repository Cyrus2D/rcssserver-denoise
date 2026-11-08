# Exp2 training and offline evaluation

Run these commands from `experiments/exp2`. The [experiment summary](../README.md) records the model selected for live use.

## Generate the dataset

From `experiments/exp2` in WSL, with NumPy and pandas installed:

~~~sh
python3 -B code/build_dataset.py
~~~

By default the builder reads `/home/nader/workspace/denoise/logs/exp1`
and writes `/home/nader/workspace/denoise/logs/exp2`. The destination
must be a new directory. Use `--source`, `--output`, and `--workers`
to override the defaults. A failed build leaves the destination unpublished.
The builder reads only `agent_cycle.csv.gz`, `landmarks.csv.gz` and
`meta.json`. It stores one memory-mapped NumPy shard per game and
a JSON index with splits, agent segment boundaries and physics settings.
Expect about 2 GB for the full dataset.

For a smoke run use `--limit-games 3 --output /tmp/exp2-dataset-smoke`.
Every game belongs to one split, including all agents on both teams.
The default split for 100 games is 80 training, 10 validation and 10
test, shuffled with seed 42. Segments stop at missed decisions, gaps,
or breaks in `play_on`. The input sequence can continue through a
cycle with missing truth, but that cycle contributes no training target.

## Train and inspect

Set up a separate Python environment. This example uses the CPU wheel,
which also provides a fallback when the GPU or driver cannot run PyTorch.

~~~sh
python3 -m venv /home/nader/workspace/denoise/.venv-exp2
source /home/nader/workspace/denoise/.venv-exp2/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -B code/train.py
~~~

CPU is the default. The Quadro P2200 ran this tracker with PyTorch
2.7.1 and CUDA 11.8, but CPU was faster on the three-game smoke dataset.
With 8,192 training windows at batch size 128 and the same 99,128
validation cycles, CPU took 4.5 s and CUDA took 14.9 s. CPU was also
faster for the training pass at batch size 512.

To try GPU training in a separate environment, install the CUDA wheel
and run with `--device cuda`. Use a new `--run-dir` for the comparison.

~~~sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -B code/train.py --device cuda --run-dir /home/nader/workspace/denoise/logs/exp2/runs/gru48-cuda
~~~

Run TensorBoard from the training environment:

~~~sh
tensorboard --logdir /home/nader/workspace/denoise/logs/exp2/runs --host 127.0.0.1
~~~

Open [TensorBoard](http://localhost:6006). Start with validation mean
and p90 against librcsc and fixed fusion. The age breakdown shows
where any improvement comes from. Use the gate histograms to spot
a model that always trusts one source. Use loss, gradient norm and
learning rate to diagnose unstable training.

Once you have chosen a checkpoint on validation, run the held-out test:

~~~sh
python -B code/train.py --evaluate /home/nader/workspace/denoise/logs/exp2/runs/gru48/best.pt
~~~

The run saves `best.pt`, the hyperparameters, validation metrics and
TensorBoard events. Checkpoint evaluation saves `metrics.json` and
test events under `evaluation-test/`. For a wiring smoke test use the
three-game dataset and train with `--epochs 1 --windows-per-epoch 512`.

To compare the position-only hybrid on validation, use the same
checkpoint with `--hybrid`:

~~~sh
python -B code/train.py --evaluate /home/nader/workspace/denoise/logs/exp2/runs/gru48/best.pt --split validation --hybrid
~~~

The comparison adds a hybrid rollout that carries its own position
and uses librcsc’s velocity and orientation at every cycle. It writes
metrics and TensorBoard events under `evaluation-validation-hybrid/`.

To compare motion physics using the type known to each agent, build a
separate index from the original raw message logs. The default command
indexes the validation split and stores it alongside the dataset.

~~~sh
python -B code/build_player_physics.py
python -B code/train.py --evaluate /home/nader/workspace/denoise/logs/exp2/runs/gru48/best.pt \
  --split validation --player-physics /home/nader/workspace/denoise/logs/exp2/player-physics-validation
~~~

This adds inertia-only and full player-type comparisons using the
same checkpoint. Metrics and events go under
`evaluation-validation-player-physics/`. Use a new `--output` or
`--run-dir` for a repeat.

## Compare individual landmarks with landmark summaries

Step 04 trains two position-only hybrids with the Step 03 motion
model. The summary variant uses the existing landmark summaries.
The flags variant adds fixed slots for the 53 flags and two goals.
Each visible slot encodes distance and bearing with a visibility mask.
The network receives the sine and cosine of the bearing.

~~~sh
python -B code/build_player_physics.py --split train
python -B code/build_landmark_slots.py
python -B code/train_landmarks.py --variant summary
python -B code/train_landmarks.py --variant flags
~~~

The two runs use the same initialization seed, game split, windows,
training budget, and validation checkpoint rule. Results and
TensorBoard events go under `runs/step04-summary` and `runs/step04-flags`.
Use a new `--run-dir` to repeat training. The builders require an unused
output path. The validation physics index from Step 03 is reused.

To score the selected checkpoints on test, prepare the separate
test indices and run:

~~~sh
python -B code/build_player_physics.py --split test
python -B code/build_landmark_slots.py --split test --output /home/nader/workspace/denoise/logs/exp2/landmark-slots-test
python -B code/train_landmarks.py --evaluate /home/nader/workspace/denoise/logs/exp2/runs/step04-summary/best.pt --split test
python -B code/train_landmarks.py --evaluate /home/nader/workspace/denoise/logs/exp2/runs/step04-flags/best.pt --split test \
  --landmarks /home/nader/workspace/denoise/logs/exp2/landmark-slots-test
~~~

## Compare training and architecture changes

Step 05 keeps the position-only hybrid, player-specific physics and
game split from Step 04. The gate diagnostic uses fullstate on
validation to compute the best scalar weight for each fresh fix,
holding the learned rollout’s motion prediction fixed. The diagnostic
score is an offline reference.

~~~sh
python -B code/evaluate_gate_oracle.py
python -B code/train_landmarks.py --variant summary --windows-per-epoch 60000 \
  --run-dir /home/nader/workspace/denoise/logs/exp2/runs/step05-baseline
~~~

Compare the baseline with one change at a time. Keep the same
seed, window schedule and eight-epoch budget, and choose a new
`--run-dir` for each option.

| Option | Change |
|---|---|
| `--hidden-size 96` | Double the recurrent width |
| `--circular-angles` | Encode dash direction, sensed movement direction and visual looking direction as sine and cosine |
| `--position-residual` | Add a learned two-dimensional position correction of up to 10 cm on a fresh fix |
| `--no-landmark-summary` | Drop the landmark summary features while keeping the visual fix and its error bounds |

The checkpoint records its input and architecture options, so
`train_landmarks.py --evaluate` restores the same model.

## What the original tracker uses

| Source | At cycle t |
|---|---|
| Motion state | Its previous estimate of position, velocity, body angle and neck angle |
| Commands | Dash, body turn and neck turn sent at t−1, with action flags |
| Body sensor | Speed, movement direction, neck angle and collision indicators |
| Visual localization | Current raw position and face fixes and their error bounds, with separate availability flags |
| Landmark geometry | Counts, measured ranges, bearing spread, and the map span of the observed flags and goals |
| Timing | View width and the ages of the position and face fixes |

The motion step predicts the current state using the previous state
and the commands. The GRU also sees the difference between the fresh
fix and that prediction. It outputs a position weight and a face
weight between zero and one. The tracker uses them to correct its
prediction when a fix is available and returns position, velocity,
body angle and looking direction. The weights describe the visual
correction, and have no effect when the corresponding fix is absent.

Fullstate is stored in a separate label field. Its coordinates and
any truth-derived error columns stay out of the inputs. The current
cycle's action is also unavailable to the predictor because it is
chosen after the estimate. The dataset uses raw visual fixes logged by
librcsc, so this experiment evaluates fusion and propagation on the
recorded sensor stream. The motion equations are approximate and use
the server's base player parameters. Individual player types,
collisions, and command execution can depart from that approximation.

## Step 01 training configuration

| Choice | Default |
|---|---|
| Network | One 48-unit GRU cell with two sigmoid gates, about 13.5k trainable parameters |
| Training sequence | 11 steps following a logged librcsc seed, 5 warm-up steps, then loss on the remaining steps |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001, gradient norm clipped to 1 |
| Training budget | 128 windows per batch, up to 60,000 sampled windows per epoch, up to 20 epochs |
| Loss | Smooth L1 on position distance and wrapped face error, scaled by 5 cm and 0.5° |
| Checkpoint | Best validation position and face score, with learning-rate reduction on a plateau and early stopping |
| Comparisons | Logged librcsc and the same tracker with fixed visual weights of 0.80 for position and 0.95 for face |

A training window is seeded once from the preceding librcsc state.
Within the window, the model carries its own estimate and memory.
Validation and test run across full contiguous segments with no
periodic reset, after the same five-step warm-up. All three estimators
are scored on the same cycles. The evaluation groups position and
face by their own visual fix ages. It also reports velocity error as a
diagnostic.

Look for lower mean and p90 on held-out cycles, especially at ages 1
and 2, while checking fresh fixes for regressions. Check both the
fixed tracker and librcsc before crediting the GRU. The offline replay
keeps the original game actions. A separate test inside the running
agent is needed to measure changes in its decisions.
