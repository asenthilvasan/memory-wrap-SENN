"""Optional Weights & Biases tracking, enabled with --wandb.

When disabled, init() returns a no-op run so callers never need to check.
"""
import absl.flags

absl.flags.DEFINE_bool("wandb", False, "Log this run to Weights & Biases.")
absl.flags.DEFINE_string("wandb_project", "memory-wrap", "W&B project name.")
absl.flags.DEFINE_string("wandb_entity", None, "W&B user or team. Defaults to your W&B default entity.")
FLAGS = absl.flags.FLAGS


class _NoOpRun:
    def __init__(self):
        self.summary = {}

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def init(name: str, group: str, job_type: str, config: dict):
    """Start a W&B run, or return a no-op run when --wandb is off.

    Args:
        name (str): Run name, e.g. 'encoder_memory_supcon-seed3'.
        group (str): Runs sharing a group are averaged together in the W&B UI.
            Use one group per ablation cell so its seeds aggregate.
        job_type (str): 'pretrain', 'train' or 'purity'.
        config (dict): Hyperparameters to record with the run.
    """
    if not FLAGS.wandb:
        return _NoOpRun()
    import wandb
    return wandb.init(project=FLAGS.wandb_project, entity=FLAGS.wandb_entity,
                      name=name, group=group, job_type=job_type, config=config)
