from vec3 import Vec3, random_double, random_int
from math import floor
class Perlin:
    POINT_COUNT = 256

    def __init__(self):
        self.randfloat = [Vec3(random_double(-1, 1), random_double(-1, 1), random_double(-1, 1)).unit() for _ in range(self.POINT_COUNT)]
        self.perm_x = self.perlin_generate_perm()
        self.perm_y = self.perlin_generate_perm()
        self.perm_z = self.perlin_generate_perm()

    def noise(self, p: Vec3) -> float:
        # Python's int() truncates toward zero and `&` acts on two's complement, like C++,
        # so negative coordinates land on the same lattice cells.
        i = int(floor(p.x)) & 255
        j = int(floor(p.y)) & 255
        k = int(floor(p.z)) & 255
        u = p.x - floor(p.x)
        v = p.y - floor(p.y)
        w = p.z - floor(p.z)
        # Use hermitian smoothing for the interpolation weights
 
        c = [[[0.0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    ii = (i + di) & 255
                    jj = (j + dj) & 255
                    kk = (k + dk) & 255
                    # Use ii, jj, kk as needed for interpolation or other calculations
                    c[di][dj][dk] = self.randfloat[self.perm_x[ii] ^ self.perm_y[jj] ^ self.perm_z[kk]]

        return self.perlin_interp(c, u, v, w)

    @classmethod
    def perlin_generate_perm(cls) -> list[int]:
        # C++ fills a caller-owned array; returning a new list is the Python equivalent.
        p = list(range(cls.POINT_COUNT))
        cls.permute(p, cls.POINT_COUNT)
        return p

    @staticmethod
    def permute(p: list[int], n: int):
        # Fisher-Yates shuffle, in place.
        for i in range(n - 1, 0, -1):
            target = random_int(0, i)
            p[i], p[target] = p[target], p[i]

    @staticmethod 
    def tri_interp(c: list[list[list[float]]], u: float, v: float, w: float) -> float:
        accum = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    accum += (i * u + (1 - i) * (1 - u)) * \
                             (j * v + (1 - j) * (1 - v)) * \
                             (k * w + (1 - k) * (1 - w)) * c[i][j][k]
        return accum

    @staticmethod
    def perlin_interp(c: list[list[list[Vec3]]], u: float, v: float, w: float) -> float:
        # Hermitian smoothing
        uu = u * u * (3 - 2 * u)
        vv = v * v * (3 - 2 * v)
        ww = w * w * (3 - 2 * w)
        accum = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    weight_v = Vec3(u - i, v - j, w - k)
                    accum += ((i * uu + (1 - i) * (1 - uu)) *
                              (j * vv + (1 - j) * (1 - vv)) *
                              (k * ww + (1 - k) * (1 - ww)) *
                              c[i][j][k].dot(weight_v))
        return accum

    def turbulance(self, p: Vec3, depth: int = 7) -> float:
        accum = 0.0
        temp_p = p
        weight = 1.0
        for _ in range(depth):
            accum += weight * self.noise(temp_p)
            weight *= 0.5
            temp_p = temp_p * 2.0
        return abs(accum)