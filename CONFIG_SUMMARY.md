# Configuration summary

Description of all configuration parameters organized in the sub-configs.
Some configurations **must** be overwritten in experiment config file depending on
the context of use. For instance to use conditioning model you should specified
`config.model.architecture='cond_sagan'` as well as other configuration
described in "Conditioning" sections below.

Moreover, the most important configuration parameters that impact
performance and may be fitted for your dataset and machine in priority are
indicated with an asterisk (\*) at the beginning of the line. **All parameters
of "Conditioning" sections are important for conditional methods but ignored in
unconditional models.**

## Main

Default configuration in `gan_facies/configs/default/main.yaml`.
Associated name space: `config.*`.

- (*)`run_name`: the name of the run.
  **Must be overwritten in experiment config file**

- (*)`config_save_path`: the path where the full configuration will be saved.
  **Must be overwritten in experiment config file**

- (*)`dataset_path`: the path of the dataset to use (in .npy uint8 format)
  **Must be overwritten in experiment config file**

- `output_dir` (default 'res'): output "global" directory. For a given experiment,
  the path of the result will be `output_dir/run_name`.

- `seed` (default 0): the seed of the run for stochastic operation (be aware
  that the reproducibility is not warranty as the cudnn seed is not controlled)

- (*)`recover_model_step` (default 0): the number of the step to recover the models.
  The path of the model to use are determined by the name of the run and
  this parameter. To do not recover model, use recover_model_step = 0.
  This parameter is also used for test script and if it is not specified,
  the script looks for the last saved model.

- `save_attn` (default False): whether the attention matrices will be saved
  (in .npy format) or not during training and testing.

- `trunc_ampl` (default -1): truncation trick. The random input is sampled from
  a truncated normal distribution with norm below `trunc_ampl` (if > 0)
  **for metric calculation or test only!**. Set it to non positive to disable

- `{*}_config_path`: additional paths for default configuration.
  **You should not overwrite it** unless you know what you are doing.

## Data

Default configuration in `gan_facies/configs/default/data.yaml`.
Associated name space: `config.data.*`.

**Note that the data size (= resolution) is also a property of the model and
it is in model configuration (see below).**

- (*)`train_batch_size` (default 64): batch size for training

- `test_batch_size` (default 64): batch size for testing (if below 64,
  the grid of sampled generated images will be lower than 64)

- `num_workers` (default 0): if > 0, it is the number of worker to use in
  parallel for data pre-processing and loading

- `shuffle` (default True): whether the data are shuffled at end of epoch or not

- `prefetch_factor` (default 2): number of batch to load with CPU while
  the GPU is running

- `persistant_workers` (default False): whether copying worker at the end of epoch
  (False) or keep them (True)

- `pin_memory` (default False): whether to pin memory on GPU for faster transfer
  or not

### Conditioning

- `n_pixels_cond`: number of pixel to sample to condition the model (if int).
  If it is a list of two integers, the number of sample is uniformly get between
  the two integers (and can be different for each sample in each batch).
  **Must be overwritten in conditional model.**

- `pixel_size_cond` (default 6): size of the pixels to condition with

- `pixel_classes_cond` (default []): list of classes of the pixels to
  condition with. If empty, all classes are used.

## Model

Default configuration in `gan_facies/configs/default/model.yaml`.
Associated name space: `config.model.*`.

- (*)`architecture` (default 'sagan'): the name of the architecture.
  Currently 'sagan' and 'cond_sagan' are implemented.
  **You must overwrite this parameter to 'cond_sagan' to "activate" the conditioning**

- (*)`data_size` (default 64): data size (or equivalently resolution) of the images
  to generate. The shape is always square. Should be either 32, 64, 128 or 256.

- `init_method` (default 'default'): initialization method. Could be either
  'default' (default initialization of Pytorch), 'orthogonal', 'normal' or 'glorot'.

- (*)`attn_layer_num` (default [3, 4]). The layer index (starting from 1) where the
  self-attention will be applied on the outputs. Keep it empty to remove self-attention.

- `sym_attn` (default True). The generator and the discriminator has similar architecture
  but reversed. If true, the indexes of layers with self-attention for the discriminator
  will be `num_layers - attn_layer_num` to preserve symmetric reversed architecture.
  If false, the indexes are `attn_layer_num` for both models.

- `z_dim` (default 128): dimension of the random input

- (*)`g_conv_dim`/`d_conv_dim` (default 64): number of channels before the last
  layer of the generator and number of channels after the first layer of the
  discriminator. Concretely, the depth of the models at each layer is
  proportional to these parameters.

### Conditioning

- `cond_dim_ratio` (default -1): ratio between number of non-conditional channels
  and conditional channels. Basically, 8 is a good ratio.
  **Must be overwritten in conditional model.**

### Attention

Note: these parameters are in the name space `config.model.attention.*`.

- `n_heads` (default 1): number of attention head to use

- `out_layer` (default True): whether to apply an output layer or not. If False,
  the dimension of values should be the input dimension so `v_ratio = 1` otherwise
  an error will be raised.

- `qk_ratio` (default 8): ratio between input dimension and queries/keys dimension

- `v_ratio` (default 2): ratio between input dimension and value dimension

## Training

Default configuration in `gan_facies/configs/default/training.yaml`.
Associated name space: `config.training.*`.

- `interrupt_threshod` (default -1): stop the training if the sum of the
  absolute losses are above this value (negative to disable). Convenient for
  automatic search in learning rates space to stop the training early.
  Negative to disable.

- `save_boxes` (default True): whether to save plot boxes of indicators distributions
  during training

- `adv_loss` (default 'wgan-gp'): loss of GAN. Should be 'wgan-gp' or 'hinge'.
  WGAN-GP is recommended to avoid mode collapse.

- `mixed_precision` (default False): whether to use mixed precision to save
  memory during training and maybe reduce training time.

- `g_ema_decay`/`d_ema_decay` (default 1.0): decay for Exponential Moving
  Average (EMA) in parameters. EMA is disabled if `*_ema_decay` = 1.0.

- `d_iters` (default 1): number of discriminator updates for each generator update.

- (*)`g_lr`/`d_lr` (default 0.0001 and 0.0004): learning rates.

- `lambda_gp` (default 10): weight for gradient penalty term of the loss.
  Only used if `adv_loss` is 'wgan_gp'.

- `beta1`/`beta2` (default 0.0 and 0.9): betas parameters for Adam optimizer

- `weight_decay` (default 0.0): weight decay (disabled if 0.0)

### Numbers of steps/iterations (1 step = 1 models update)

- (*)`total_step` (default 100,000): total number of step for training

- `total_time` (default -1): total maximum time in second for training (if positive)

- `ema_start_step` (default 0): step for which the EMA starts (only used if
  the ema decays are different from 1.0)

- `log_step` (default 10): number of steps between every log in console

- `sample_step` (default 400): number of steps between every image generation
  (and saving)

- `model_save_step` (default 1200): number of steps between every model saving

- `metric_step` (default 1200): number of steps between every metric calculation

### Conditioning

- `cond_penalty` (default 10): weight for conditional loss.
  Only used in conditional models.

## Metrics

Default configuration in `gan_facies/configs/default/metrics.yaml`.
Associated name space: `config.metrics.*`.

- `overwrite_indicators` (default False): whether re-computing indicators distribution
  from ground truth images even if they are already computed.

- `connectivity` (default None): connectivity between cells for metric computation.
  If 1, only the orthogonal neighbor are considered as connected.
  If 2, the diagonal are also considered. If 3 (only for 3D models),
  the corners are also considered. If None, the connectivity is inferred from
  dimension of the model (2 for 2D models and 3 for 3D models).

- (*)`unit_component_size` (default 6): maximal number of pixels in a connected
  component to consider it as "unit component" for metric calculation.

## Distributed

Default configuration in `gan_facies/configs/default/distributed.yaml`.
Associated name space: `config.distributed.*`.

These parameters are passed in `ignite.distributed.launcher.Parallel`.
See the *torch ignite* or *pytorch.distributed* documentation for more details.

- `backend` (default 'nccl'): backend for gradient distribution

- `nnodes` (default 1): number of machine to use.
  **Must be overwritten to use multiple machines**

- `nproc_per_node` (default 1): number of GPUs per machine.
  **Must be overwritten to use multiple GPUs**

- `init_method` (default None): URL specifying how to initialize the process group.
  If None, the default method is used.

- `node_rank`/`master_addr`/`master_port` (default None): specs of the master machine.
  **Should be overwritten to use multiple machines (but not for multiple GPUs
  on a single machine)**

## Experiment tracking

Default configuration in `gan_facies/configs/default/experiment_tracking.yaml`.
Associated name space: `config.experiment_tracking.*`.

Note: you can't use Wandb and ClearML simultaneously.

### Weights and Biases

- `use_wandb` (default False): whether to use wandb or not

- `project`: name of the project. **Must be overwritten if wandb is used**

- `entity` (default None): name of the wandb account (if None, the default
  account is used)

- `group` (default None): name of the group of experiment (if None, a new random
  group is created)

- `mode` (default 'online'): whether used wandb online (default usage)
  or offline (experiments will be sent to wandb after the first connection)

- `id` (default None): used to resume an experiment

#### Sweep

- `sweep_config_path` (default None): path to sweep configuration (more details
  in *wandb.sweep* documentation).

- `sweep` (default None): sub-config that will contain the
  sweep configuration. **Do not overwrite**

### ClearML

- `use_clearml` (default False): whether to use clearml or not

- `project_name`: name of the project. **Must be overwritten if clearml is used**

- `task_name`: name of the experiment. **Must be overwritten if clearml is used**

- `tags` (default None): list of tags for the experiment (if
  None, no tags are used)

- `continue_last_task` (default None): whether to resume a previous task.
  If True, the last task is resumed. Can also be an experiment index to resume
  an older task. If False or None, a new task is created.
