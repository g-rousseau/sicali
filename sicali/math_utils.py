import math


def wrap_to_pi(angle: float) -> float:
    """Wrap the input in ]-pi, pi]."""
    res = math.fmod(angle, 2 * math.pi)
    if res < 0.0:
        res += 2 * math.pi

    if res > math.pi:
        res -= 2 * math.pi

    return res
