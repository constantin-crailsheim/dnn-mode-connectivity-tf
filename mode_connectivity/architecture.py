from dataclasses import dataclass
from typing import Any, Dict, List, Type

import tensorflow as tf


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
