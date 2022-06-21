from mode_connectivity.curves import CurveLayer
import pytest


class TestCurveLayer:
    def test_init_direct(self):
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class CurveLayer with abstract methods build, call",
        ):
            CurveLayer([True, False, True])
