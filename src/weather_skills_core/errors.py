"""SkillError / UsageError (exit 2) / DataError (exit 1)."""


class SkillError(Exception):
    """Base CLI error. prefix=False prints the message without 'Error: '."""

    exit_code = 1

    def __init__(self, *args, prefix: bool = True):
        super().__init__(*args)
        self.prefix = prefix


class UsageError(SkillError):
    """Usage/validation failure; exit 2."""

    exit_code = 2


class DataError(SkillError):
    """Data/runtime failure; exit 1."""

    exit_code = 1
