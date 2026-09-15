from __future__ import annotations
from typing import TYPE_CHECKING
from vec3 import Vec3
from dataclasses import dataclass
from ray import Ray
from abc import ABC, abstractmethod
from math import acos, atan2, cos, pi, radians, sin, sqrt
import numpy as np

if TYPE_CHECKING:
    from material import Material

@dataclass
class HitRecord:
    t: float
    p: Vec3
    normal: Vec3
    material: Material
    u: float
    v: float
    front_face: bool = False

    def set_face_normal(self, r: Ray, outward_normal: Vec3):
        # Sets the hit record normal vector.
        # NOTE: the parameter `outward_normal` is assumed to have unit length.
        self.front_face = r.direction.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal

class Hittable(ABC):
    @abstractmethod
    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord | None:
        pass
    @abstractmethod
    def bounding_box(self) -> AABB | None:
        pass



@dataclass
class Quad(Hittable):
    Q: Vec3
    u: Vec3
    v: Vec3
    material: Material

    def __post_init__(self):
        n = self.u.cross(self.v)
        self.normal = n.unit()
        self.D = self.normal.dot(self.Q)
        # w uses the unnormalized n, so alpha/beta come out in units of u and v.
        self.w = n / n.dot(n)

    def bounding_box(self) -> AABB | None:
        bbox_diagonal1 = AABB.from_points(self.Q, self.Q + self.u + self.v)
        bbox_diagonal2 = AABB.from_points(self.Q + self.u, self.Q + self.v)
        return AABB.from_boxes(bbox_diagonal1, bbox_diagonal2)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        # Plain float math, like Sphere.hit: most calls miss, so avoid creating Vec3s until a hit.
        n, origin, direction = self.normal, ray.origin, ray.direction
        dx, dy, dz = direction.x, direction.y, direction.z

        # No hit if the ray is parallel to the plane.
        denom = n.x * dx + n.y * dy + n.z * dz
        if abs(denom) < 1e-8:
            return None

        # No hit if the hit point parameter t is outside the ray interval.
        t = (self.D - (n.x * origin.x + n.y * origin.y + n.z * origin.z)) / denom
        if not (t_min < t < t_max):
            return None

        # Determine if the hit point lies within the planar shape using its plane coordinates.
        px = origin.x + t * dx
        py = origin.y + t * dy
        pz = origin.z + t * dz
        Q, u, v, w = self.Q, self.u, self.v, self.w
        hx, hy, hz = px - Q.x, py - Q.y, pz - Q.z

        # alpha = w . (h x v); reject before computing beta.
        alpha = w.x * (hy * v.z - hz * v.y) + w.y * (hz * v.x - hx * v.z) + w.z * (hx * v.y - hy * v.x)
        if not 0.0 <= alpha <= 1.0:
            return None
        # beta = w . (u x h)
        beta = w.x * (u.y * hz - u.z * hy) + w.y * (u.z * hx - u.x * hz) + w.z * (u.x * hy - u.y * hx)
        if not Quad.is_interior(alpha, beta):
            return None

        # Same test as set_face_normal: denom is direction . normal.
        front_face = denom < 0
        return HitRecord(t=t, p=Vec3(px, py, pz), normal=n if front_face else -n,
                         material=self.material, u=alpha, v=beta, front_face=front_face)

    @staticmethod
    def is_interior(a: float, b: float) -> bool:
        # Given the hit point in plane coordinates, return whether it lies inside the quad.
        # The (u, v) texture coordinates are just (a, b), so hit() sets them directly.
        return 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0


@dataclass
class Sphere(Hittable):
    center: Vec3
    radius: float
    material: Material

    def bounding_box(self) -> AABB | None:
        rvec = Vec3(self.radius, self.radius, self.radius)
        bbox = AABB.from_points(self.center - rvec, self.center + rvec)
        return bbox

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        # Plain float math: this runs for every ray/sphere pair, so avoid creating Vec3s on a miss.
        origin, direction, center = ray.origin, ray.direction, self.center
        dx, dy, dz = direction.x, direction.y, direction.z
        ocx = origin.x - center.x
        ocy = origin.y - center.y
        ocz = origin.z - center.z

        a = dx * dx + dy * dy + dz * dz
        half_b = ocx * dx + ocy * dy + ocz * dz
        c = ocx * ocx + ocy * ocy + ocz * ocz - self.radius * self.radius

        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return None

        sqrt_discriminant = sqrt(discriminant)

        # Try the nearer intersection first.
        root = (-half_b - sqrt_discriminant) / a
        if not (t_min < root < t_max):
            # The ray may start inside the sphere; try the farther intersection.
            root = (-half_b + sqrt_discriminant) / a
            if not (t_min < root < t_max):
                return None

        # Still floats: only the final point and normal become Vec3s.
        px = origin.x + root * dx
        py = origin.y + root * dy
        pz = origin.z + root * dz
        radius = self.radius
        nx = (px - center.x) / radius
        ny = (py - center.y) / radius
        nz = (pz - center.z) / radius
        outward_normal = Vec3(nx, ny, nz)
        # UVs come from the outward normal: flipping it would mirror the texture on inside hits.
        u, v = Sphere.get_sphere_uv(outward_normal)

        # Same test as set_face_normal.
        front_face = dx * nx + dy * ny + dz * nz < 0
        return HitRecord(t=root, p=Vec3(px, py, pz), normal=outward_normal if front_face else -outward_normal,
                         material=self.material, u=u, v=v, front_face=front_face)

    @staticmethod
    def get_sphere_uv(p: Vec3) -> tuple[float, float]:
        # p: a point on the unit sphere centered at the origin.
        # Python can't write through float arguments like C++'s u/v references, so (u, v) is returned:
        # u: value [0,1] of angle around the Y axis from X=-1.
        # v: value [0,1] of angle from Y=-1 to Y=+1.
        theta = acos(-p.y)
        phi = atan2(-p.z, p.x) + pi
        return phi / (2 * pi), theta / pi
@dataclass
class MovingSphere(Hittable):
    # Center moves linearly from center0 at time 0 to center1 at time 1.
    center0: Vec3
    center1: Vec3
    radius: float
    material: Material

    def bounding_box(self) -> AABB | None:
        rvec = Vec3(self.radius, self.radius, self.radius)
        bbox1 = AABB.from_points(self.center0 - rvec, self.center0 + rvec)
        bbox2 = AABB.from_points(self.center1 - rvec, self.center1 + rvec)
        bbox = AABB.from_boxes(bbox1, bbox2)
        return bbox

    def center(self, time: float) -> Vec3:
        return self.center0 + time * (self.center1 - self.center0)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        origin, direction, center = ray.origin, ray.direction, self.center(ray.time)
        dx, dy, dz = direction.x, direction.y, direction.z
        ocx = origin.x - center.x
        ocy = origin.y - center.y
        ocz = origin.z - center.z

        a = dx * dx + dy * dy + dz * dz
        half_b = ocx * dx + ocy * dy + ocz * dz
        c = ocx * ocx + ocy * ocy + ocz * ocz - self.radius * self.radius

        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return None

        sqrt_discriminant = sqrt(discriminant)

        root = (-half_b - sqrt_discriminant) / a
        if not (t_min < root < t_max):
            root = (-half_b + sqrt_discriminant) / a
            if not (t_min < root < t_max):
                return None

        p = ray.at(root)
        outward_normal = (p - center) / self.radius
        u, v = Sphere.get_sphere_uv(outward_normal)

        record = HitRecord(t=root, p=p, normal=outward_normal, material=self.material, u=u, v=v)
        record.set_face_normal(ray, outward_normal)
        return record


class HittableList(Hittable):
    def __init__(self):
        self.objects: list[Hittable] = []
        self.bbox = AABB()

    def add(self, obj: Hittable):
        self.objects.append(obj)
        self.bbox = AABB.from_boxes(self.bbox, obj.bounding_box())

    def bounding_box(self) -> AABB | None:
        return self.bbox

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        hit_record: HitRecord | None = None
        closest_so_far = t_max

        for obj in self.objects:
            temp_record = obj.hit(ray, t_min, closest_so_far)
            if temp_record:
                closest_so_far = temp_record.t
                hit_record = temp_record

        return hit_record

class Box(HittableList):
    def __init__(self, a: Vec3, b: Vec3, material: Material):
        super().__init__()
        self.min = Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
        self.max = Vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))
        dx = Vec3(self.max.x - self.min.x, 0.0, 0.0)
        dy = Vec3(0.0, self.max.y - self.min.y, 0.0)
        dz = Vec3(0.0, 0.0, self.max.z - self.min.z)
        self.add(Quad(Vec3(self.min.x, self.min.y, self.max.z), dx, dy, material))   # front
        self.add(Quad(Vec3(self.max.x, self.min.y, self.max.z), -dz, dy, material))  # right
        self.add(Quad(Vec3(self.max.x, self.min.y, self.min.z), -dx, dy, material))  # back
        self.add(Quad(Vec3(self.min.x, self.min.y, self.min.z), dz, dy, material))   # left
        self.add(Quad(Vec3(self.min.x, self.max.y, self.max.z), dx, -dz, material))  # top
        self.add(Quad(Vec3(self.min.x, self.min.y, self.min.z), dx, dz, material))   # bottom

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        # One box test is much cheaper than six quad tests, and most rays that get here miss.
        if not self.bbox.hit(ray, t_min, t_max):
            return None
        return super().hit(ray, t_min, t_max)

class Interval:
    # The default interval is empty (min > max).
    def __init__(self, t_min: float = float("inf"), t_max: float = float("-inf")):
        self.t_min = t_min
        self.t_max = t_max

    def size(self) -> float:
        return self.t_max - self.t_min

    def expand(self, delta: float):
        padding = delta/2
        return Interval(self.t_min - padding, self.t_max + padding)
    
    def clamp(self, x: float) -> float:
        if x < self.t_min:
            return self.t_min
        if x > self.t_max:
            return self.t_max
        return x
    
    @staticmethod
    def surrounding_interval(a: Interval, b: Interval) -> Interval:
        return Interval(min(a.t_min, b.t_min), max(a.t_max, b.t_max))

    def __add__(self, displacement: float) -> Interval:
        return Interval(self.t_min + displacement, self.t_max + displacement)

    # float.__add__ returns NotImplemented for an Interval, so `displacement + ival` lands here.
    __radd__ = __add__


Interval.EMPTY = Interval(float("inf"), float("-inf"))
Interval.UNIVERSE = Interval(float("-inf"), float("inf"))

class AABB:
    # The default AABB is empty, since intervals are empty by default.
    def __init__(
        self,
        x_interval: Interval | None = None,
        y_interval: Interval | None = None,
        z_interval: Interval | None = None,
    ):
        self.x_interval = x_interval if x_interval is not None else Interval()
        self.y_interval = y_interval if y_interval is not None else Interval()
        self.z_interval = z_interval if z_interval is not None else Interval()

    @classmethod
    def from_points(cls, a: Vec3, b: Vec3) -> AABB:
        # Treat the two points a and b as extrema for the bounding box, so we don't require a
        # particular minimum/maximum coordinate order. Pad to minimus before returning

        return cls(
            Interval(min(a.x, b.x), max(a.x, b.x)),
            Interval(min(a.y, b.y), max(a.y, b.y)),
            Interval(min(a.z, b.z), max(a.z, b.z)),
        ).pad_to_minimus()
    
    def pad_to_minimus(self, delta: float = 0.00001):
        if self.x_interval.size() < delta:
            self.x_interval = self.x_interval.expand(delta)
        if self.y_interval.size() < delta:
            self.y_interval = self.y_interval.expand(delta)
        if self.z_interval.size() < delta:
            self.z_interval = self.z_interval.expand(delta)
        return self

    @classmethod
    def from_boxes(cls, box1: AABB, box2: AABB) -> AABB:
        return cls(
            Interval.surrounding_interval(box1.x_interval, box2.x_interval),
            Interval.surrounding_interval(box1.y_interval, box2.y_interval),
            Interval.surrounding_interval(box1.z_interval, box2.z_interval),
        )
        
    def axis_interval(self, n: int) -> Interval:
        if n == 1:
            return self.y_interval
        if n == 2:
            return self.z_interval
        return self.x_interval

    def hit(self, ray: Ray, t_min: float, t_max: float) -> bool:
        # This runs dozens of times per ray, so the three axes are written out by hand
        # instead of looping over axis_interval(): no loop, no method calls, no indexing.
        origin, direction = ray.origin, ray.direction

        # Python raises on 1.0 / 0.0 (C++ gives inf): a ray parallel to a slab
        # misses unless its origin lies between the slab planes.
        ax = self.x_interval
        orig, d = origin.x, direction.x
        if d == 0.0:
            if orig < ax.t_min or orig > ax.t_max:
                return False
        else:
            adinv = 1.0 / d
            t0 = (ax.t_min - orig) * adinv
            t1 = (ax.t_max - orig) * adinv
            if t0 > t1:
                t0, t1 = t1, t0
            if t0 > t_min:
                t_min = t0
            if t1 < t_max:
                t_max = t1
            if t_max <= t_min:
                return False

        ax = self.y_interval
        orig, d = origin.y, direction.y
        if d == 0.0:
            if orig < ax.t_min or orig > ax.t_max:
                return False
        else:
            adinv = 1.0 / d
            t0 = (ax.t_min - orig) * adinv
            t1 = (ax.t_max - orig) * adinv
            if t0 > t1:
                t0, t1 = t1, t0
            if t0 > t_min:
                t_min = t0
            if t1 < t_max:
                t_max = t1
            if t_max <= t_min:
                return False

        ax = self.z_interval
        orig, d = origin.z, direction.z
        if d == 0.0:
            if orig < ax.t_min or orig > ax.t_max:
                return False
        else:
            adinv = 1.0 / d
            t0 = (ax.t_min - orig) * adinv
            t1 = (ax.t_max - orig) * adinv
            if t0 > t1:
                t0, t1 = t1, t0
            if t0 > t_min:
                t_min = t0
            if t1 < t_max:
                t_max = t1
            if t_max <= t_min:
                return False

        return True

    def longest_axis(self) -> int:
        # Returns the index of the longest axis of the bounding box.
        x_size = self.x_interval.size()
        y_size = self.y_interval.size()
        z_size = self.z_interval.size()

        if x_size > y_size:
            return 0 if x_size > z_size else 2
        else:
            return 1 if y_size > z_size else 2

    def __add__(self, offset: Vec3) -> AABB:
        return AABB(self.x_interval + offset.x, self.y_interval + offset.y, self.z_interval + offset.z)

AABB.EMPTY = AABB(Interval.EMPTY, Interval.EMPTY, Interval.EMPTY)
AABB.UNIVERSE = AABB(Interval.UNIVERSE, Interval.UNIVERSE, Interval.UNIVERSE)

class BVH_Node(Hittable):
    def __init__(self, objects: list[Hittable], start: int = 0, end: int | None = None):
        if end is None:
            end = len(objects)

        # Build the bounding box of the span of source objects.
        self.bbox = AABB.EMPTY
        for object_index in range(start, end):
            self.bbox = AABB.from_boxes(self.bbox, objects[object_index].bounding_box())

        # Split along the longest axis: halves separated along the widest spread overlap least.
        axis = self.bbox.longest_axis()

        comparator = (BVH_Node.box_x_compare if axis == 0
                      else BVH_Node.box_y_compare if axis == 1
                      else BVH_Node.box_z_compare)

        # hit() uses the split axis to visit the nearer child first.
        self.axis = axis

        object_span = end - start

        if object_span == 1:
            self.left = self.right = objects[start]
        else:
            # Spans of two are sorted too, so left is always the lower child along the axis.
            objects[start:end] = sorted(objects[start:end], key=comparator)
            mid = start + object_span // 2
            # A half holding a single object is stored directly: wrapping it in a node would
            # cost an extra box test and hit the same object twice (left = right).
            self.left = objects[start] if mid - start == 1 else BVH_Node(objects, start, mid)
            self.right = objects[mid] if end - mid == 1 else BVH_Node(objects, mid, end)

    # Python's sort takes a key rather than a less-than comparator, so these return the value
    # to order by (the box's minimum on that axis) instead of comparing two objects.
    @staticmethod
    def box_compare(obj: Hittable, axis_index: int) -> float:
        return obj.bounding_box().axis_interval(axis_index).t_min

    @staticmethod
    def box_x_compare(obj: Hittable) -> float:
        return BVH_Node.box_compare(obj, 0)

    @staticmethod
    def box_y_compare(obj: Hittable) -> float:
        return BVH_Node.box_compare(obj, 1)

    @staticmethod
    def box_z_compare(obj: Hittable) -> float:
        return BVH_Node.box_compare(obj, 2)

    @classmethod
    def from_list(cls, hittable_list: HittableList) -> BVH_Node:
        # Sort a copy so the caller's list keeps its order (mirrors the C++ copy by value).
        return cls(list(hittable_list.objects))

    def bounding_box(self) -> AABB | None:
        return self.bbox

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        if not self.bbox.hit(ray, t_min, t_max):
            return None

        # Visit the child nearer the ray first: left holds the lower coordinates on the split
        # axis, so a ray travelling in the negative direction reaches the right child first.
        # Finding the closest hit early shrinks t_max, which lets the other child's box be
        # rejected more often.
        d = ray.direction
        axis = self.axis
        if (d.x if axis == 0 else d.y if axis == 1 else d.z) < 0.0:
            first, second = self.right, self.left
        else:
            first, second = self.left, self.right

        hit_first = first.hit(ray, t_min, t_max)
        # Only accept a hit from the second child that is closer than the first one.
        hit_second = second.hit(ray, t_min, hit_first.t if hit_first is not None else t_max)

        return hit_second if hit_second is not None else hit_first

class Translate(Hittable):
    def __init__(self, object: Hittable, offset: Vec3):
        self.object = object
        self.offset = offset
        self.bbox = object.bounding_box() + offset

    def bounding_box(self) -> AABB | None:
        return self.bbox

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        # Move the ray backwards by the offset.
        offset_r = Ray(ray.origin - self.offset, ray.direction, ray.time)

        # Determine whether an intersection exists along the offset ray (and if so, where).
        record = self.object.hit(offset_r, t_min, t_max)
        if record is None:
            return None

        # Move the intersection point forwards by the offset.
        record.p = record.p + self.offset
        return record

class RotateY(Hittable):
    def __init__(self, object: Hittable, angle: float):
        self.object = object
        theta = radians(angle)
        self.sin_theta = sin(theta)
        self.cos_theta = cos(theta)
        bbox = object.bounding_box()

        # Vec3 has no item assignment, so the running extrema are plain lists.
        min_pt = [float("inf")] * 3
        max_pt = [float("-inf")] * 3

        # Rotate all eight corners of the object's box and bound the results.
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = i * bbox.x_interval.t_max + (1 - i) * bbox.x_interval.t_min
                    y = j * bbox.y_interval.t_max + (1 - j) * bbox.y_interval.t_min
                    z = k * bbox.z_interval.t_max + (1 - k) * bbox.z_interval.t_min

                    newx = self.cos_theta * x + self.sin_theta * z
                    newz = -self.sin_theta * x + self.cos_theta * z

                    tester = (newx, y, newz)
                    for c in range(3):
                        min_pt[c] = min(min_pt[c], tester[c])
                        max_pt[c] = max(max_pt[c], tester[c])

        self.bbox = AABB.from_points(Vec3(*min_pt), Vec3(*max_pt))

    def bounding_box(self) -> AABB | None:
        return self.bbox

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        cos_theta, sin_theta = self.cos_theta, self.sin_theta

        # Transform the ray from world space to object space.
        o, d = ray.origin, ray.direction
        origin = Vec3(cos_theta * o.x - sin_theta * o.z, o.y, sin_theta * o.x + cos_theta * o.z)
        direction = Vec3(cos_theta * d.x - sin_theta * d.z, d.y, sin_theta * d.x + cos_theta * d.z)
        rotated_r = Ray(origin, direction, ray.time)

        # Determine whether an intersection exists in object space (and if so, where).
        record = self.object.hit(rotated_r, t_min, t_max)
        if record is None:
            return None

        # Transform the intersection from object space back to world space.
        p, n = record.p, record.normal
        record.p = Vec3(cos_theta * p.x + sin_theta * p.z, p.y, -sin_theta * p.x + cos_theta * p.z)
        record.normal = Vec3(cos_theta * n.x + sin_theta * n.z, n.y, -sin_theta * n.x + cos_theta * n.z)
        return record
