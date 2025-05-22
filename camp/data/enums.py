from enum import StrEnum, auto


class LowerStrEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name.lower()


class PreferenceRegionType(LowerStrEnum):
    CLUSTER = auto()
    PI = auto()
    RANDOM = auto()


class ObjectiveType(LowerStrEnum):
    MINSUM = auto()
    MINMAX = auto()


class CostMatrixType(LowerStrEnum):
    Distance = auto()
    Duration = auto()
