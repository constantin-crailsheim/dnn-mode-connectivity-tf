import tensorflow as tf
from typing import Any, List, Type, Dict
from dataclasses import dataclass


class CurveModel(tf.keras.Model):
    fix_points: List[bool]

    def __init__(self, fix_points: List[bool], *args, **kwargs) -> None:
        self.fix_points = fix_points
        super().__init__(*args, **kwargs)


@dataclass
class Architecture:
    base: Type[tf.keras.Model]
    curve: Type[CurveModel]
    kwargs: Dict[Any, Any]
