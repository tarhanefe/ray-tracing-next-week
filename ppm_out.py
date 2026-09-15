from multiprocessing import Pool

import numpy as np

from vec3 import Vec3
from ray import Ray
from hitable import Hittable, HittableList, Sphere
from material import Lambertian, Metal, Dielectric
from camera import Camera

def random_in_unit_sphere() -> Vec3:
    while True:
        p = 2.0 * Vec3(np.random.rand(), np.random.rand(), np.random.rand()) - Vec3(1.0, 1.0, 1.0)
        if p.length_squared() < 1.0:
            return p

def color(ray: Ray, world: Hittable, depth: int) -> Vec3:
    record = world.hit(ray, 0.001, float("inf"))

    if record is not None:
        if depth <= 0:
            return Vec3(0.0, 0.0, 0.0)

        scatter_result = record.material.scatter(ray, record)

        if scatter_result is None:
            return Vec3(0.0, 0.0, 0.0)

        attenuation, scattered_ray = scatter_result

        bounced_color = color(scattered_ray, world, depth - 1)

        return attenuation.hadamard(bounced_color)

    # The ray missed every object: draw the sky.
    unit_direction = ray.direction.unit()
    t = 0.5 * (unit_direction.y + 1.0)

    return (
        (1.0 - t) * Vec3(1.0, 1.0, 1.0)
        + t * Vec3(0.5, 0.7, 1.0)
    )


# Set once per worker process, so the scene isn't re-sent with every row.
_scene = None

def init_worker(camera, world, width, height, samples_per_pixel) -> None:
    global _scene
    _scene = (camera, world, width, height, samples_per_pixel)
    # Forked workers would otherwise inherit identical NumPy RNG state.
    np.random.seed()

def render_row(i: int) -> np.ndarray:
    camera, world, width, height, samples_per_pixel = _scene
    row = np.zeros((width, 3), dtype=np.uint8)

    for j in range(width):
        pixel_color = Vec3(0.0, 0.0, 0.0)
        for s in range(samples_per_pixel):
            u = (j + np.random.rand()) / width
            v = (height - 1 - i + np.random.rand()) / height

            ray = camera.get_ray(u, v)
            pixel_color += color(ray, world, 50)

        pixel_color /= samples_per_pixel
        pixel_color = Vec3(np.sqrt(pixel_color.r), np.sqrt(pixel_color.g), np.sqrt(pixel_color.b))
        row[j] = [
            int(255.99 * pixel_color.r),
            int(255.99 * pixel_color.g),
            int(255.99 * pixel_color.b),
        ]

    return row


def print_ppm() -> None:
    aspect_ratio = 16.0 / 9.0
    width = 400
    height = int(width / aspect_ratio)
    samples_per_pixel = 100
    ppm = np.zeros((height, width, 3), dtype=np.uint8)

    camera = Camera(
        lookfrom=Vec3(-2.0, 2.0, 1.0),
        lookat=Vec3(0.0, 0.0, -1.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=20.0,
        focus_dist=3.4,
        defocus_angle=10.0,
    )
    
    world = HittableList()
    world.add(Sphere(Vec3(0.0, 0.0, -1.2), 0.5, Lambertian(Vec3(0.1, 0.2, 0.5))))
    world.add(Sphere(Vec3(0.0, -100.5, -1.0), 100.0, Lambertian(Vec3(0.8, 0.8, 0.0))))
    world.add(Sphere(Vec3(1.0, 0.0, -1.0), 0.5, Metal(Vec3(0.8, 0.6, 0.2), fuzz=1)))
    world.add(Sphere(Vec3(-1.0, 0.0, -1.0), 0.5, Dielectric(1.5)))
    world.add(Sphere(Vec3(-1.0, 0.0, -1.0), 0.4, Dielectric(1/1.5)))

    initargs = (camera, world, width, height, samples_per_pixel)
    with Pool(initializer=init_worker, initargs=initargs) as pool:
        # imap yields rows in order, as soon as each one is ready.
        for i, row in enumerate(pool.imap(render_row, range(height))):
            ppm[i] = row
            print(f"\rRows done: {i + 1}/{height}", end="", flush=True)
    print()

    # P6 is a compact binary PPM format.
    with open("output.ppm", "wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode())
        file.write(ppm.tobytes())


if __name__ == "__main__":
    print_ppm()