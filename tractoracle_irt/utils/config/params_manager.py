import os
from tractoracle_irt.utils.config.misc import is_false

class ParamsManager:
    def __init__(self, separator=" ", prefix="--", indent_nb_spaces=4):
        self._flags = []
        self.separator = separator
        self.prefix = prefix
        self.indentation = indent_nb_spaces * " "
        self.script_path = None

    def compile_command(self):
        compiled_flags = self.compile_flags(linebreak=True, start_with_linebreak=True, indent=1)

        command = [
            f"python -O {self.script_path} {compiled_flags}"
        ]
        return "\n".join(command)

    def register_script(self, script_path):
        """
        Register a script to be executed.
        This is typically used to specify the main script for the experiment.
        """
        if not script_path:
            raise ValueError("Script path cannot be None or empty.")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script '{script_path}' does not exist.")
        
        self.script_path = script_path

    def add_pos_arg(self, value):
        self.add_param(value, skip_if_none=False, no_prefix=True)
        
    def add_param(self, flag, value=None, skip_if_none=True, required=False, no_prefix=False):
        if flag is None:
            raise ValueError("Flag cannot be None.")
        
        if required and value is None:
            raise ValueError(f"Flag '{flag}' is required but has no value.")

        if value is None and skip_if_none:
            return

        if flag.endswith("_") and value is None:
            return # Skip flags that end with an underscore and have no value

        if is_false(value):
            return # Skip flags that are False
        
        if not flag.startswith(self.prefix) and not no_prefix:
            flag = self.prefix + flag

        self._flags.append(str(flag))

        if value is not None \
            and not isinstance(value, bool):
            self._flags.append(str(value))

    def add_flag_if_true(self, flag, value):
        if is_false(value):
            return
        
        if not flag.startswith(self.prefix):
            prefixed_flag = self.prefix + flag
        else:
            prefixed_flag = flag
        
        self._flags.append(prefixed_flag)

    def add_this_or_that(self, flag1, value1, flag2, value2, require_one=True):
        """
        Add one of two flags based on their values.
        If value1 is not None, add flag1 with value1.
        If value2 is not None, add flag2 with value2.
        If both are None, do nothing.
        """

        if value1 is not None and not is_false(value1):
            self.add_flag(flag1, value1)
        elif value2 is not None and not is_false(value2):
            self.add_flag(flag2, value2)
        elif require_one:
            raise ValueError(f"Must specify one of {flag1} or {flag2}, but both are False/None.")

    def add_config(self, extras_config: dict):
        for key, value in extras_config.items():
            self.add_flag(key, value)

    def compile_flags(self, linebreak=False, start_with_linebreak=False, indent=0):
        if linebreak:
            start = ""
            sep = "\\\n" + (self.indentation * indent)
            if start_with_linebreak:
                start += sep
            return start + sep.join(self._flags)
        else:
            return self.separator.join(self._flags)
    
    def __str__(self):
        return self.compile_flags(linebreak=False)