from abc import ABC, abstractmethod
from math import sqrt
from dataclasses import dataclass
from ray import Ray
from vec3 import Vec3
from hitable import HitRecord
import random
from texture import *

class Material(ABC):
    @abstractmethod
    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> tuple[Vec3, Ray] | None:
        pass

    def emitted(self, u: float, v: float, p: Vec3) -> Vec3:
        # Most materials give off no light of their own.
        return Vec3(0.0, 0.0, 0.0)

def random_in_unit_sphere() -> Vec3:
    while True:
        p = 2.0 * Vec3(random.random(), random.random(), random.random()) - Vec3(1.0, 1.0, 1.0)
        if p.length_squared() < 1.0:
            return p

def random_unit_vector() -> Vec3:
    while True:
        p = 2.0 * Vec3(random.random(), random.random(), random.random()) - Vec3(1.0, 1.0, 1.0)
        length_squared = p.length_squared()
        # Reject points outside the sphere, and points so close to the center that
        # normalizing them would underflow to zero.
        if 1e-160 < length_squared <= 1.0:
            return p / sqrt(length_squared)

class Lambertian(Material):
    def __init__(self, albedo: Vec3 | Texture):
        # One constructor for both C++ ones: a plain color is wrapped in a SolidColor.
        self.tex = albedo if isinstance(albedo, Texture) else SolidColor(albedo)

    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> tuple[Vec3, Ray]:
        scatter_direction = record.normal + random_unit_vector()

        # Catch degenerate scatter direction
        if scatter_direction.near_zero():
            scatter_direction = record.normal

        scattered_ray = Ray(record.p, scatter_direction, time=incoming_ray.time)
        attenuation = self.tex.value(record.u, record.v, record.p)

        return attenuation, scattered_ray

def reflect(v: Vec3, n: Vec3) -> Vec3:
    return v - 2.0 * v.dot(n) * n

def refract(v: Vec3, normal: Vec3, eta: float) -> Vec3 | None:
    """
    v: incoming ray direction
    normal: surface normal pointing against the incoming ray
    eta: refractive-index ratio, n1 / n2
    """
    unit_v = v.unit()

    cos_theta = min((-unit_v).dot(normal), 1.0)
    perpendicular = eta * (unit_v + cos_theta * normal)

    parallel_length_squared = 1.0 - perpendicular.length_squared()

    # No refracted ray is possible.
    if parallel_length_squared < 0.0:
        return None

    parallel = -sqrt(parallel_length_squared) * normal

    return perpendicular + parallel

def reflectance(cosine: float, refraction_ratio: float) -> float:
    r0 = ((1.0 - refraction_ratio) / (1.0 + refraction_ratio)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cosine) ** 5

@dataclass
class Metal(Material):
    albedo: Vec3
    fuzz: float = 0.0

    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> tuple[Vec3, Ray] | None:
        reflected = reflect(
            incoming_ray.direction.unit(),
            record.normal,
        )

        scattered_ray = Ray(record.p, reflected + self.fuzz * random_in_unit_sphere(), time=incoming_ray.time)

        # A valid reflection must leave the surface.
        if scattered_ray.direction.dot(record.normal) <= 0:
            return None

        return self.albedo, scattered_ray

@dataclass
class Dielectric(Material):
    refraction_index: float  # Glass is usually 1.5

    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> tuple[Vec3, Ray]:
        # Glass is clear: it does not tint or absorb light here.
        attenuation = Vec3(1.0, 1.0, 1.0)

        unit_direction = incoming_ray.direction.unit()

        # record.normal already points against the ray; front_face says which side was hit.
        normal = record.normal
        # Entering: air -> glass. Leaving: glass -> air.
        eta = 1.0 / self.refraction_index if record.front_face else self.refraction_index

        cos_theta = min((-unit_direction).dot(normal), 1.0)
        sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

        # Total internal reflection, or probabilistic normal reflection.
        must_reflect = eta * sin_theta > 1.0

        if must_reflect or reflectance(cos_theta, eta) > random.random():
            direction = reflect(unit_direction, normal)
        else:
            direction = refract(unit_direction, normal, eta)

        scattered_ray = Ray(record.p, direction, time=incoming_ray.time)

        return attenuation, scattered_ray

class DiffuseLight(Material):
    def __init__(self, emit: Vec3 | Texture):
        # One constructor for both C++ ones: a plain color is wrapped in a SolidColor.
        self.tex = emit if isinstance(emit, Texture) else SolidColor(emit)

    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> None:
        # A light only emits; it never bounces rays.
        return None

    def emitted(self, u: float, v: float, p: Vec3) -> Vec3:
        return self.tex.value(u, v, p)

class Isotropic(Material):
    def __init__(self, albedo: Vec3 | Texture):
        # One constructor for both C++ ones: a plain color is wrapped in a SolidColor.
        self.tex = albedo if isinstance(albedo, Texture) else SolidColor(albedo)

    def scatter(
        self,
        incoming_ray: Ray,
        record: HitRecord,
    ) -> tuple[Vec3, Ray]:
        # Scatter in a uniformly random direction, ignoring the (arbitrary) normal.
        scattered_ray = Ray(record.p, random_unit_vector(), time=incoming_ray.time)
        attenuation = self.tex.value(record.u, record.v, record.p)
        return attenuation, scattered_ray
