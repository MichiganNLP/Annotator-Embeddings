import pytorch_lightning as pl
from overrides import overrides


class CustomizedModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        self.whether_ease_constraint = kwargs.pop("whether_ease_constraint", False)
        super().__init__(*args, **kwargs)

    @overrides
    def _should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        if not self.whether_ease_constraint:
            return (
                trainer.fast_dev_run  # disable checkpointing with fast_dev_run
                or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
                or trainer.sanity_checking  # don't save anything during sanity check
                or self._last_global_step_saved == trainer.global_step  # already saved at the last step
            )
        return (
            trainer.fast_dev_run  # disable checkpointing with fast_dev_run
            or trainer.state.fn not in {TrainerFn.FITTING, TrainerFn.VALIDATING}  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or self._last_global_step_saved == trainer.global_step  # already saved at the last step
        )