from dataclasses import dataclass
from math import sqrt
import numpy as np

@dataclass 
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def r(self): return self.x
    @property
    def g(self): return self.y
    @property
    def b(self): return self.z

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, other: float) -> "Vec3":
        return Vec3(self.x * other, self.y * other, self.z * other)

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> "Vec3":
        return Vec3(self.x / other, self.y / other, self.z / other)
    
    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]

    def near_zero(self) -> bool:
        # Return true if the vector is close to zero in all dimensions.
        s = 1e-8
        return abs(self.x) < s and abs(self.y) < s and abs(self.z) < s

    def length_squared(self) -> float:
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def length(self) -> float:
        return sqrt(self.length_squared())

    def unit(self) -> "Vec3":
        return self / self.length()

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self,other:"Vec3"):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    def hadamard(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
    )

    @staticmethod
    def random(min: float = 0.0, max: float = 1.0) -> "Vec3":
        # Each component is a random real in [min,max).
        return Vec3(random_double(min, max), random_double(min, max), random_double(min, max))

def random_double(min: float = 0.0, max: float = 1.0) -> float:
    # Returns a random real in [min,max).
    return min + (max - min) * np.random.rand()

def random_int(min: int, max: int) -> int:
    # Returns a random integer in [min,max].
    return int(random_double(min, max + 1))

def random_in_unit_disk() -> "Vec3":
    while True:
        p = 2.0 * Vec3(np.random.rand(), np.random.rand(), 0.0) - Vec3(1.0, 1.0, 0.0)
        if p.length_squared() < 1.0:
            return p