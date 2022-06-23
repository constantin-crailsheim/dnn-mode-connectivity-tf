from mode_connectivity.curves.curves import Bezier, Curve
import numpy as np


class TestBezier:
    def test_init(self):
        curve = Bezier(3)
        assert isinstance(curve, Bezier)
        assert isinstance(curve, Curve)

    def test_call(self):
        curve = Bezier(3)
        t = 0.5  # Input between 0-1
        output_1 = curve(t)
        output_2 = curve(t)
        output_3 = curve(t)
        assert np.allclose(output_1, output_2, output_3)
