import numpy as np

from vec3 import Vec3, random_in_unit_disk
from ray import Ray

class Camera:
    def __init__(
        self,
        lookfrom: Vec3 = Vec3(0.0, 0.0, 0.0),   # Point camera is looking from
        lookat: Vec3 = Vec3(0.0, 0.0, -1.0),    # Point camera is looking at
        up: Vec3 = Vec3(0.0, 1.0, 0.0),         # Camera-relative "up" direction
        vfov: float = 90.0,                     # Vertical field of view in degrees
        aspect_ratio: float = 16.0 / 9.0,       # Ratio of image width over height
        focus_dist: float = 10.0,               # Distance from lookfrom to plane of perfect focus
        defocus_angle: float = 0.0,             # Variation angle of rays through each pixel
        background: Vec3 = Vec3(0.0, 0.0, 0.0),  # Background color
    ):
        self.center = lookfrom
        self.focus_dist = focus_dist
        self.defocus_angle = defocus_angle
        self.background = background

        # Determine viewport dimensions. The viewport lies on the focus plane.
        h = np.tan(np.radians(vfov) / 2.0)
        viewport_height = 2.0 * h * focus_dist
        viewport_width = aspect_ratio * viewport_height

        # Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        self.w = (lookfrom - lookat).unit()
        self.u = up.cross(self.w).unit()
        self.v = self.w.cross(self.u)

        # Vectors across the full horizontal (left -> right) and vertical (bottom -> top) viewport edges.
        self.horizontal = viewport_width * self.u
        self.vertical = viewport_height * self.v

        # Calculate the location of the lower left corner of the viewport.
        self.lower_left_corner = (
            self.center
            - focus_dist * self.w
            - self.horizontal / 2
            - self.vertical / 2
        )

        # Calculate the camera defocus disk basis vectors.
        defocus_radius = focus_dist * np.tan(np.radians(defocus_angle / 2.0))
        self.defocus_disk_u = defocus_radius * self.u
        self.defocus_disk_v = defocus_radius * self.v

    def get_ray(self, s: float, t: float) -> Ray:
        """Ray from the defocus disk through viewport point (s, t).

        s, t in [0, 1]: s goes left -> right, t goes bottom -> top.
        """
        viewport_point = self.lower_left_corner + s * self.horizontal + t * self.vertical

        ray_origin = self.center if self.defocus_angle <= 0 else self.defocus_disk_sample()
        ray_direction = viewport_point - ray_origin
        # rand() over uniform(0, 1): same [0, 1) draw, and it is called once per camera ray.
        ray_time = np.random.rand()
        return Ray(ray_origin, ray_direction, ray_time)

    def defocus_disk_sample(self) -> Vec3:
        """Returns a random point in the camera defocus disk."""
        p = random_in_unit_disk()
        return self.center + p.x * self.defocus_disk_u + p.y * self.defocus_disk_v
