# Visual experiments for "Learning From the Past with Cascading Eligibility Traces"

## Installation

To run the experiments, install the required packages:

```bash
pip install -r requirements.txt
```

## Experiments

- `delay` is the number of images between input presentation and the arrival of credit signals
  - run for `(1,3,5,10,20,50)` for behavioural timescales and `600` for retrograde timescale (2 minutes)
- `ssm_degree` is the CET order minus 1
  - run for `(0,1,5,9)` for ET, CET(2),CET(6),CET(10)

### Behavioural timescales

#### CIFAR 10
 
```bash

python btsp_experiments.py \
		--batch_size=128 \
		--effective_batch=128 \
		--hidden_dim=512 \
		--training_steps=$training_steps \
		--ssm_degree="$degree" \
		--delay_factor="$delay" \
		--save_folder=$folder \
		--run_name="run_name" \
		--n_groups=1 \
		--interp_factor=1 \
		--top_quantile=0.0 \
		--sparse_inputs \
		--adam \
		--dataset="cifar10" \
		--cnn \
		--wandb_project="btsp_cifar_final_configs" \
		--norm_op="batch" \
		--weight_decay="$weight_decay" \
		--p_drop=0.0 \
		--wandb_mode="online" \
		--warmup=0.1 \
		--track_sim
```

#### MNIST

```bash
python btsp_experiments.py \
        --batch_size=128 \
        --effective_batch=128 \
        --hidden_dim=512 \
        --training_steps=$training_steps \
        --ssm_degree="$degree" \
        --delay_factor="$delay" \
        --save_folder=$folder \
        --run_name="run_name" \
        --n_groups=1 \
        --interp_factor=1 \
        --top_quantile=0.0 \
        --sparse_inputs \
        --adam \
        --dataset="mnist" \
        --wandb_project="wandb_project" \
        --weight_decay="$weight_decay" \
        --lr="$lr" \
        --p_drop=0.0 \
        --wandb_mode="online" \
        --warmup=0.1 \
        --track_sim
```

### Retrograde timescales

#### CIFAR 10

```bash

 python fixed_sparse.py \
        --batch_size=1280 \
        --effective_batch=1280 \
        --hidden_dim=512 \
        --training_steps=$training_steps \
        --ssm_degree="$degree" \
        --delay_factor="$delay" \
        --save_folder=$folder \
        --run_name="run_name" \
        --n_groups=1 \
        --interp_factor=1 \
        --top_quantile=0.9875 \
        --sparse_inputs \
        --adam \
        --dataset="cifar10" \
        --cnn \
        --wandb_project="wandb_project" \
        --norm_op="batch" \
        --weight_decay="$weight_decay" \
        --p_drop=0.0 \
        --lr="$lr" \
        --min_lr=0.1 \
        --val_steps=3125 \
        --warmup=0.2 \ 
		--track_sim
```

#### MNIST

```bash
python fixed_sparse.py \
		--batch_size=1280 \
		--effective_batch=1280 \
		--hidden_dim=512 \
		--training_steps=$training_steps \
		--ssm_degree="$degree" \
		--delay_factor="$delay_factor" \
		--save_folder=$folder \
		--run_name="$run_name" \
		--n_groups=1 \
		--interp_factor=1 \
		--top_quantile=0.9875 \
		--sparse_inputs \
		--adam \
		--dataset="mnist" \
		--wandb_project="wandb_project" \
		--norm_op="$norm" \
		--weight_decay="$weight_decay" \
		--p_drop=0.0 \
		--lr="$lr" \
		--min_lr=0.1 \
		--warmup=0.2 \
		--track_sim
```
