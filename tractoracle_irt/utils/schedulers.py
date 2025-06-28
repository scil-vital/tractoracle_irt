from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

class ValueScheduler(object):
    """
    Abstract class for a value scheduling mechanism.
    """
    def __init__(self):
        pass

    @property
    def value(self):
        assert False, "Not implemented"

    def step(self):
        assert False, "Not implemented"

    def reset(self):
        return self

class LinearInterpolationScheduler(ValueScheduler):
    """
    This controller interpolates between two values linearly.
    """
    def __init__(self, base_value, end_value, initial_step, final_step):
        LOGGER.info("Initializing LinearInterpolationScheduler from {} to {} "
                    "between steps {} and {}".format(base_value, end_value,
                                                     initial_step, final_step))
        self.base_value = base_value
        self.end_value = end_value
        self.initial_step = initial_step
        self.final_step = final_step

        self.total_nb_steps = final_step - initial_step
        self.current_step = 0

    @property
    def value(self):
        if self.current_step < self.initial_step:
            return self.base_value
        if self.current_step > self.final_step:
            return self.end_value
        
        lmda = (self.current_step - self.initial_step) / self.total_nb_steps
        return (1 - lmda) * self.base_value + lmda * self.end_value
        
    def step(self):
        self.current_step += 1

    def reset(self):
        self.current_step = 0
        return self

class ConstantScheduler(ValueScheduler):
    """
    This controller keeps the value constant.
    """
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    def step(self):
        pass