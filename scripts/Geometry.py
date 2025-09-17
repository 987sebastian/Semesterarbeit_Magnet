from dataclasses import dataclass


@dataclass
class CylinderGeom:
    Br: float  # remanence [T] / 剩磁
    R: float  # radius [m] / 半径
    l: float  # length [m] / 长度
