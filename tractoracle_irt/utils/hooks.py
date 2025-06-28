
from enum import Enum


class HookEvent(Enum):
    pass


class OracleHookEvent(HookEvent):
    ON_TRAIN_EPOCH_START = 'on_train_epoch_start'
    ON_TRAIN_EPOCH_END = 'on_train_epoch_end'
    ON_TRAIN_BATCH_START = 'on_train_batch_start'
    ON_TRAIN_MICRO_BATCH_START = 'on_train_micro_batch_start'
    ON_TRAIN_MICRO_BATCH_END = 'on_train_micro_batch_end'
    ON_TRAIN_BATCH_END = 'on_train_batch_end'
    ON_VAL_BATCH_START = 'on_val_batch_start'
    ON_VAL_BATCH_END = 'on_val_batch_end'
    ON_TEST_START = 'on_test_start'
    ON_TEST_END = 'on_test_end'


class RlHookEvent(HookEvent):
    ON_RL_TRAIN_START = 'on_rl_train_start'
    ON_RL_BEST_VC = 'on_rl_best_vc'
    ON_RL_TRAIN_END = 'on_rl_train_end'


class HooksManager(object):
    def __init__(self, hooks_enum: type[HookEvent]) -> None:
        self._hooks = {event: [] for event in hooks_enum}

    def register_hook(self, event, hook):
        self._hooks[event].append(hook)

    def unregister_hook(self, event, hook):
        self._hooks[event].remove(hook)

    def trigger_hooks(self, event, *args, **kwargs):
        for hook in self._hooks[event]:
            hook(*args, **kwargs)
