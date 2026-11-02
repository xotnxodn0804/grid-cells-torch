from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    env_size: float = 2.2
    n_pc: tuple = (256,)
    pc_scale: tuple = (0.01,)
    n_hdc: tuple = (12,)
    hdc_concentration: tuple = (20.0,)
    neurons_seed: int = 0
    velocity_inputs: bool = True
    velocity_noise: tuple = (0.0, 0.0, 0.0)


@dataclass
class DataGenConfig:
    duration_s: float = 15.0
    sim_dt: float = 0.02
    perimeter_m: float = 0.03
    speed_rayleigh_scale: float = 0.13
    rot_vel_std_deg: float = 330.0
    perimeter_speed_reduction: float = 0.25
    n_steps: int = 100
    n_shards: int = 100
    trajectories_per_shard: int = 10_000
    seed: int = 0


@dataclass
class ModelConfig:
    nh_lstm: int = 128
    nh_bottleneck: int = 512
    dropout_rates: tuple = (0.5,)
    weight_decay: float = 1e-5
    weight_decay_scope: str = "output_heads"
    bottleneck_has_bias: bool = True
    ego_vel_dim: int = 3


@dataclass
class TrainConfig:
    epochs: int = 300
    steps_per_epoch: int = 1000
    minibatch_size: int = 10
    learning_rate: float = 1e-5
    momentum: float = 0.9
    grad_clip_value: float = 1e-5
    grad_clip_scope: str = "output_heads"
    save_every_n_epochs: int = 20
    results_dir: str = "data/checkpoints/baseline"
    seed: int = 0


@dataclass
class RLEnvConfig:
    level: str = "square_arena"
    level_directory: str = "levels"
    width: int = 64
    height: int = 64
    grid_size: int = 10
    env_size_m: float = 2.5
    wall_world_units: tuple = (100.0, 1100.0)
    action_repeat: int = 4
    num_actors: int = 32
    velocity_noise_std: float = 0.01
    goal_visible: bool = True


@dataclass
class VisionRLConfig:
    embed_dim: int = 256
    mask_prob: float = 0.95
    learning_rate: float = 1e-4
    minibatch_size: int = 32


@dataclass
class GridRLConfig:
    minibatch_size: int = 10
    seq_len: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class ActorCriticConfig:
    lstm_units: int = 256
    embed_dim: int = 256
    learning_rate_range: tuple = (0.000001, 0.0002)
    gradient_momentum: float = 0.99
    discount: float = 0.99
    entropy_reg_range: tuple = (0.00006, 0.0001)
    baseline_cost_range: tuple = (0.48, 0.52)
    n_step: int = 100


@dataclass
class A2CConfig:
    num_envs: int = 32
    device: str = "cuda"
    vision_updates_per_rollout: int = 51
    grid_updates_per_rollout: int = 2
    replay_frames: int = 400_000
    vision_mask_mode: str = "per_unit"
    reset_policy_on_goal: bool = False
    cut_return_on_goal: bool = False
    reinit_grid_on_goal: bool = False
    grid_train_zero_init: bool = True
    ac_grad_clip_norm: float | None = None
    ac_rmsprop_alpha: float = 0.99
    ac_rmsprop_eps: float = 1e-8
    ac_rmsprop_momentum: float | None = 0.0
    checkpoint_every_frames: int = 10_000_000
    renderer: str = "hardware"


@dataclass
class ReplayBufferConfig:
    frame_capacity: int = 200_000
    sequence_capacity: int = 2_000


@dataclass
class RLConfig:
    env: RLEnvConfig = field(default_factory=RLEnvConfig)
    vision: VisionRLConfig = field(default_factory=VisionRLConfig)
    grid: GridRLConfig = field(default_factory=GridRLConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)
    pc_scale: tuple = (0.4,)
    total_env_steps: int = 1_000_000_000
    seed: int = 0
    results_dir: str = "data/checkpoints/rl_baseline"
    checkpoint_every_env_steps: int = 500_000
    log_every_n_updates: int = 20
    resume_from: str | None = None


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataGenConfig = field(default_factory=DataGenConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rl: RLConfig = field(default_factory=RLConfig)
