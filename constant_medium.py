from __future__ import annotations
from math import log
from hitable import AABB, HitRecord, Hittable
from material import Isotropic, Material
from ray import Ray
from texture import Texture
from vec3 import Vec3, random_double

class ConstantMedium(Hittable):
    def __init__(self, boundary: Hittable, density: float, albedo: Vec3 | Texture):
        # One constructor for both C++ ones: Isotropic accepts a color or a texture.
        self.boundary = boundary
        self.neg_inv_density = -1.0 / density
        self.phase_function: Material = Isotropic(albedo)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        # Find where the ray enters and leaves the boundary, over the whole line.
        rec1 = self.boundary.hit(ray, float("-inf"), float("inf"))
        if rec1 is None:
            return None

        rec2 = self.boundary.hit(ray, rec1.t + 0.0001, float("inf"))
        if rec2 is None:
            return None

        # Clip the inside segment to the requested interval.
        t1 = max(rec1.t, t_min)
        t2 = min(rec2.t, t_max)

        if t1 >= t2:
            return None

        if t1 < 0:
            t1 = 0.0

        ray_length = ray.direction.length()
        distance_inside_boundary = (t2 - t1) * ray_length
        # random_double() is in [0, 1) and Python's log(0) raises (C++ gives -inf),
        # so sample from (0, 1] instead: same distribution.
        hit_distance = self.neg_inv_density * log(1.0 - random_double())

        if hit_distance > distance_inside_boundary:
            return None

        t = t1 + hit_distance / ray_length
        return HitRecord(
            t=t,
            p=ray.at(t),
            normal=Vec3(1.0, 0.0, 0.0),  # arbitrary
            material=self.phase_function,
            u=0.0,
            v=0.0,
            front_face=True,  # also arbitrary
        )

    def bounding_box(self) -> AABB | None:
        return self.boundary.bounding_box()
