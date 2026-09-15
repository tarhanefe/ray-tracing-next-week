from multiprocessing import Pool

import numpy as np

from vec3 import Vec3, random_double
from ray import Ray
from hitable import BVH_Node, Hittable, HittableList, MovingSphere, Quad, Sphere, Box, RotateY, Translate
from material import Lambertian, Metal, Dielectric, DiffuseLight
from camera import Camera
from texture import CheckerTexture, ImageTexture, NoiseTexture
from constant_medium import ConstantMedium

def random_in_unit_sphere() -> Vec3:
    while True:
        p = 2.0 * Vec3(np.random.rand(), np.random.rand(), np.random.rand()) - Vec3(1.0, 1.0, 1.0)
        if p.length_squared() < 1.0:
            return p

def color(ray: Ray, world: Hittable, depth: int, background: Vec3) -> Vec3:
    # If we've exceeded the ray bounce limit, no more light is gathered.
    if depth <= 0:
        return Vec3(0.0, 0.0, 0.0)

    record = world.hit(ray, 0.001, float("inf"))

    # If the ray hits nothing, return the background color.
    if record is None:
        return background

    color_from_emission = record.material.emitted(record.u, record.v, record.p)

    scatter_result = record.material.scatter(ray, record)

    if scatter_result is None:
        return color_from_emission

    attenuation, scattered_ray = scatter_result

    color_from_scatter = attenuation.hadamard(color(scattered_ray, world, depth - 1, background))

    return color_from_emission + color_from_scatter


# Set once per worker process, so the scene isn't re-sent with every row.
_scene = None

def init_worker(camera, world, width, height, samples_per_pixel, max_depth) -> None:
    global _scene
    _scene = (camera, world, width, height, samples_per_pixel, max_depth)
    # Forked workers would otherwise inherit identical NumPy RNG state.
    np.random.seed()

def render_row(i: int) -> np.ndarray:
    camera, world, width, height, samples_per_pixel, max_depth = _scene
    row = np.zeros((width, 3), dtype=np.uint8)

    for j in range(width):
        pixel_color = Vec3(0.0, 0.0, 0.0)
        for s in range(samples_per_pixel):
            u = (j + np.random.rand()) / width
            v = (height - 1 - i + np.random.rand()) / height

            ray = camera.get_ray(u, v)
            pixel_color += color(ray, world, max_depth, camera.background)

        pixel_color /= samples_per_pixel
        pixel_color = Vec3(np.sqrt(pixel_color.r), np.sqrt(pixel_color.g), np.sqrt(pixel_color.b))
        # Lights make components exceed 1, which would overflow a uint8: clamp to [0, 0.999].
        row[j] = [
            int(256 * min(max(pixel_color.r, 0.0), 0.999)),
            int(256 * min(max(pixel_color.g, 0.0), 0.999)),
            int(256 * min(max(pixel_color.b, 0.0), 0.999)),
        ]

    return row


def render(
    camera: Camera,
    world: Hittable,
    aspect_ratio: float,
    width: int,
    samples_per_pixel: int,
    max_depth: int,
) -> None:
    height = int(width / aspect_ratio)
    ppm = np.zeros((height, width, 3), dtype=np.uint8)

    initargs = (camera, world, width, height, samples_per_pixel, max_depth)
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


def bouncing_spheres() -> None:
    aspect_ratio = 16.0 / 9.0

    camera = Camera(
        lookfrom=Vec3(13.0, 2.0, 3.0),
        lookat=Vec3(0.0, 0.0, 0.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=20.0,
        focus_dist=10.0,
        defocus_angle=0.6,
        background=Vec3(0.70, 0.80, 1.00),
    )
    
    world = HittableList()
    ground_material = Lambertian(Vec3(0.5, 0.5, 0.5))
    world.add(Sphere(Vec3(0.0, -1000, .0), 1000.0, ground_material))

    for m in range(-11, 11):
        for n in range(-11, 11):
            choose_mat = np.random.rand()
            center = Vec3(m + 0.9 * np.random.rand(), 0.2, n + 0.9 * np.random.rand())
            if (center - Vec3(4.0, 0.2, 0.0)).length() > 0.9:
                if choose_mat < 0.8:
                    # diffuse
                    albedo = Vec3(np.random.rand(), np.random.rand(), np.random.rand())
                    sphere_material = Lambertian(albedo)
                    world.add(Sphere(center, 0.2, sphere_material))
                    center2 = center + Vec3(0,np.random.rand() * 0.5, 0)
                    world.add(MovingSphere(center, center2, 0.2, sphere_material))
                elif choose_mat < 0.95:
                    # metal
                    albedo = Vec3(0.5 * (1 + np.random.rand()), 0.5 * (1 + np.random.rand()), 0.5 * (1 + np.random.rand()))
                    fuzz = 0.5 * np.random.rand()
                    sphere_material = Metal(albedo, fuzz)
                    world.add(Sphere(center, 0.2, sphere_material))
                    
                else:
                    # glass
                    sphere_material = Dielectric(1.5)
                    world.add(Sphere(center, 0.2, sphere_material))
    material1 = Dielectric(1.5)
    world.add(Sphere(Vec3(0.0, 1.0, 0.0), 1.0, material1))
    material2 = Lambertian(Vec3(0.4, 0.2, 0.1))
    world.add(Sphere(Vec3(-4.0, 1.0, 0.0), 1.0, material2))
    material3 = Metal(Vec3(0.7, 0.6, 0.5), fuzz=0.0)
    world.add(Sphere(Vec3(4.0, 1.0, 0.0), 1.0, material3))

    world = BVH_Node.from_list(world)

    render(camera, world, aspect_ratio, width=1200, samples_per_pixel=100, max_depth=50)


def checkered_spheres() -> None:
    world = HittableList()

    # One texture shared by both spheres, so the checks line up where they meet.
    checker = CheckerTexture.from_colors(0.32, Vec3(255/255, 234/255, 50/255), Vec3(2/255, 57/255, 113/255))

    world.add(Sphere(Vec3(0.0, -10.0, 0.0), 10.0, Lambertian(checker)))
    world.add(Sphere(Vec3(0.0, 10.0, 0.0), 10.0, Lambertian(checker)))

    aspect_ratio = 16.0 / 9.0

    camera = Camera(
        lookfrom=Vec3(13.0, 2.0, 3.0),
        lookat=Vec3(0.0, 0.0, 0.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=20.0,
        defocus_angle=0.0,
        background=Vec3(0.70, 0.80, 1.00),
    )

    render(camera, world, aspect_ratio, width=1200, samples_per_pixel=100, max_depth=50)


def earth() -> None:
    earth_texture = ImageTexture.from_file("earthmap.jpg")
    earth_surface = Lambertian(earth_texture)
    globe = Sphere(Vec3(0.0, 0.0, 0.0), 2.0, earth_surface)

    world = HittableList()
    world.add(globe)

    aspect_ratio = 16.0 / 9.0

    camera = Camera(
        lookfrom=Vec3(0.0, 0.0, 12.0),
        lookat=Vec3(0.0, 0.0, 0.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=20.0,
        defocus_angle=0.0,
        background=Vec3(0.70, 0.80, 1.00),
    )

    render(camera, world, aspect_ratio, width=1200, samples_per_pixel=100, max_depth=50)


def perlin_spheres() -> None:
    world = HittableList()

    # One texture shared by both spheres, so they sample the same noise field.
    pertext = NoiseTexture(scale=4.0)
    world.add(Sphere(Vec3(0.0, -1000.0, 0.0), 1000.0, Lambertian(pertext)))
    world.add(Sphere(Vec3(0.0, 2.0, 0.0), 2.0, Lambertian(pertext)))

    aspect_ratio = 16.0 / 9.0

    camera = Camera(
        lookfrom=Vec3(13.0, 2.0, 3.0),
        lookat=Vec3(0.0, 0.0, 0.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=20.0,
        defocus_angle=0.0,
        background=Vec3(0.70, 0.80, 1.00),
    )

    render(camera, world, aspect_ratio, width=1200, samples_per_pixel=100, max_depth=50)


def quads() -> None:
    world = HittableList()

    # Materials
    left_red = Lambertian(Vec3(1.0, 0.2, 0.2))
    back_green = Lambertian(Vec3(0.2, 1.0, 0.2))
    right_blue = Lambertian(Vec3(0.2, 0.2, 1.0))
    upper_orange = Lambertian(Vec3(1.0, 0.5, 0.0))
    lower_teal = Lambertian(Vec3(0.2, 0.8, 0.8))

    # Quads
    world.add(Quad(Vec3(-3.0, -2.0, 5.0), Vec3(0.0, 0.0, -4.0), Vec3(0.0, 4.0, 0.0), left_red))
    world.add(Quad(Vec3(-2.0, -2.0, 0.0), Vec3(4.0, 0.0, 0.0), Vec3(0.0, 4.0, 0.0), back_green))
    world.add(Quad(Vec3(3.0, -2.0, 1.0), Vec3(0.0, 0.0, 4.0), Vec3(0.0, 4.0, 0.0), right_blue))
    world.add(Quad(Vec3(-2.0, 3.0, 1.0), Vec3(4.0, 0.0, 0.0), Vec3(0.0, 0.0, 4.0), upper_orange))
    world.add(Quad(Vec3(-2.0, -3.0, 5.0), Vec3(4.0, 0.0, 0.0), Vec3(0.0, 0.0, -4.0), lower_teal))

    aspect_ratio = 1.0

    camera = Camera(
        lookfrom=Vec3(0.0, 0.0, 9.0),
        lookat=Vec3(0.0, 0.0, 0.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=80.0,
        defocus_angle=0.0,
        background=Vec3(0.70, 0.80, 1.00),
    )

    render(camera, world, aspect_ratio, width=400, samples_per_pixel=100, max_depth=50)


def simple_light() -> None:
    world = HittableList()

    pertext = NoiseTexture(scale=4.0)
    world.add(Sphere(Vec3(0.0, -1000.0, 0.0), 1000.0, Lambertian(pertext)))
    world.add(Sphere(Vec3(0.0, 2.0, 0.0), 2.0, Lambertian(pertext)))

    # Brighter than (1,1,1), so the light is strong enough to illuminate the scene.
    difflight = DiffuseLight(Vec3(4.0, 4.0, 4.0))
    world.add(Sphere(Vec3(0.0, 7.0, 0.0), 2.0, difflight))
    world.add(Quad(Vec3(3.0, 1.0, -2.0), Vec3(2.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0), difflight))

    aspect_ratio = 16.0 / 9.0

    camera = Camera(
        lookfrom=Vec3(26.0, 3.0, 6.0),
        lookat=Vec3(0.0, 2.0, 0.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=20.0,
        defocus_angle=0.0,
        background=Vec3(0.0, 0.0, 0.0),
    )

    render(camera, world, aspect_ratio, width=1200, samples_per_pixel=100, max_depth=50)


def cornell_box() -> None:
    world = HittableList()

    red = Lambertian(Vec3(0.65, 0.05, 0.05))
    white = Lambertian(Vec3(0.73, 0.73, 0.73))
    green = Lambertian(Vec3(0.12, 0.45, 0.15))
    light = DiffuseLight(Vec3(15.0, 15.0, 15.0))

    world.add(Quad(Vec3(555.0, 0.0, 0.0), Vec3(0.0, 555.0, 0.0), Vec3(0.0, 0.0, 555.0), green))
    world.add(Quad(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 555.0, 0.0), Vec3(0.0, 0.0, 555.0), red))
    world.add(Quad(Vec3(343.0, 554.0, 332.0), Vec3(-130.0, 0.0, 0.0), Vec3(0.0, 0.0, -105.0), light))
    world.add(Quad(Vec3(0.0, 0.0, 0.0), Vec3(555.0, 0.0, 0.0), Vec3(0.0, 0.0, 555.0), white))
    world.add(Quad(Vec3(555.0, 555.0, 555.0), Vec3(-555.0, 0.0, 0.0), Vec3(0.0, 0.0, -555.0), white))
    world.add(Quad(Vec3(0.0, 0.0, 555.0), Vec3(555.0, 0.0, 0.0), Vec3(0.0, 555.0, 0.0), white))

    box1: Hittable = Box(Vec3(0.0, 0.0, 0.0), Vec3(165.0, 330.0, 165.0), white)
    box1 = RotateY(box1, 15.0)
    box1 = Translate(box1, Vec3(265.0, 0.0, 295.0))
    world.add(box1)

    box2: Hittable = Box(Vec3(0.0, 0.0, 0.0), Vec3(165.0, 165.0, 165.0), white)
    box2 = RotateY(box2, -18.0)
    box2 = Translate(box2, Vec3(130.0, 0.0, 65.0))
    world.add(box2)

    world = BVH_Node.from_list(world)

    aspect_ratio = 1.0

    camera = Camera(
        lookfrom=Vec3(278.0, 278.0, -800.0),
        lookat=Vec3(278.0, 278.0, 0.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=40.0,
        defocus_angle=0.0,
        background=Vec3(0.0, 0.0, 0.0),
    )

    render(camera, world, aspect_ratio, width=600, samples_per_pixel=100, max_depth=50)


def cornell_smoke() -> None:
    world = HittableList()

    red = Lambertian(Vec3(0.65, 0.05, 0.05))
    white = Lambertian(Vec3(0.73, 0.73, 0.73))
    green = Lambertian(Vec3(0.12, 0.45, 0.15))
    light = DiffuseLight(Vec3(7.0, 7.0, 7.0))

    world.add(Quad(Vec3(555.0, 0.0, 0.0), Vec3(0.0, 555.0, 0.0), Vec3(0.0, 0.0, 555.0), green))
    world.add(Quad(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 555.0, 0.0), Vec3(0.0, 0.0, 555.0), red))
    world.add(Quad(Vec3(113.0, 554.0, 127.0), Vec3(330.0, 0.0, 0.0), Vec3(0.0, 0.0, 305.0), light))
    world.add(Quad(Vec3(0.0, 555.0, 0.0), Vec3(555.0, 0.0, 0.0), Vec3(0.0, 0.0, 555.0), white))
    world.add(Quad(Vec3(0.0, 0.0, 0.0), Vec3(555.0, 0.0, 0.0), Vec3(0.0, 0.0, 555.0), white))
    world.add(Quad(Vec3(0.0, 0.0, 555.0), Vec3(555.0, 0.0, 0.0), Vec3(0.0, 555.0, 0.0), white))

    box1: Hittable = Box(Vec3(0.0, 0.0, 0.0), Vec3(165.0, 330.0, 165.0), white)
    box1 = RotateY(box1, 15.0)
    box1 = Translate(box1, Vec3(265.0, 0.0, 295.0))

    box2: Hittable = Box(Vec3(0.0, 0.0, 0.0), Vec3(165.0, 165.0, 165.0), white)
    box2 = RotateY(box2, -18.0)
    box2 = Translate(box2, Vec3(130.0, 0.0, 65.0))

    world.add(ConstantMedium(box1, 0.01, Vec3(0.0, 0.0, 0.0)))
    world.add(ConstantMedium(box2, 0.01, Vec3(1.0, 1.0, 1.0)))
    world = BVH_Node.from_list(world)

    aspect_ratio = 1.0

    camera = Camera(
        lookfrom=Vec3(278.0, 278.0, -800.0),
        lookat=Vec3(278.0, 278.0, 0.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=40.0,
        defocus_angle=0.0,
        background=Vec3(0.0, 0.0, 0.0),
    )

    render(camera, world, aspect_ratio, width=600, samples_per_pixel=200, max_depth=50)


def final_scene(image_width: int, samples_per_pixel: int, max_depth: int) -> None:
    # Ground: a 20x20 grid of boxes with random heights.
    boxes1 = HittableList()
    ground = Lambertian(Vec3(0.48, 0.83, 0.53))

    boxes_per_side = 20
    for i in range(boxes_per_side):
        for j in range(boxes_per_side):
            w = 100.0
            x0 = -1000.0 + i * w
            z0 = -1000.0 + j * w
            y0 = 0.0
            x1 = x0 + w
            y1 = random_double(1, 101)
            z1 = z0 + w

            boxes1.add(Box(Vec3(x0, y0, z0), Vec3(x1, y1, z1), ground))

    world = HittableList()

    world.add(BVH_Node.from_list(boxes1))

    light = DiffuseLight(Vec3(7.0, 7.0, 7.0))
    world.add(Quad(Vec3(123.0, 554.0, 147.0), Vec3(300.0, 0.0, 0.0), Vec3(0.0, 0.0, 265.0), light))

    center1 = Vec3(400.0, 400.0, 200.0)
    center2 = center1 + Vec3(30.0, 0.0, 0.0)
    sphere_material = Lambertian(Vec3(0.7, 0.3, 0.1))
    world.add(MovingSphere(center1, center2, 50.0, sphere_material))

    world.add(Sphere(Vec3(260.0, 150.0, 45.0), 50.0, Dielectric(1.5)))
    world.add(Sphere(Vec3(0.0, 150.0, 145.0), 50.0, Metal(Vec3(0.8, 0.8, 0.9), 1.0)))

    # A glass sphere filled with blue smoke: the glass shell and the medium share one boundary.
    boundary: Hittable = Sphere(Vec3(360.0, 150.0, 145.0), 70.0, Dielectric(1.5))
    world.add(boundary)
    world.add(ConstantMedium(boundary, 0.2, Vec3(0.2, 0.4, 0.9)))
    # Thin mist filling the whole scene.
    boundary = Sphere(Vec3(0.0, 0.0, 0.0), 5000.0, Dielectric(1.5))
    world.add(ConstantMedium(boundary, 0.0001, Vec3(1.0, 1.0, 1.0)))

    emat = Lambertian(ImageTexture.from_file("earthmap.jpg"))
    world.add(Sphere(Vec3(400.0, 200.0, 400.0), 100.0, emat))
    pertext = NoiseTexture(scale=0.2)
    world.add(Sphere(Vec3(220.0, 280.0, 300.0), 80.0, Lambertian(pertext)))

    # A cube-shaped cluster of 1000 small spheres, rotated and moved into place.
    boxes2 = HittableList()
    white = Lambertian(Vec3(0.73, 0.73, 0.73))
    ns = 1000
    for _ in range(ns):
        boxes2.add(Sphere(Vec3.random(0, 165), 10.0, white))

    world.add(Translate(RotateY(BVH_Node.from_list(boxes2), 15.0), Vec3(-100.0, 270.0, 395.0)))
    world = BVH_Node.from_list(world)

    aspect_ratio = 1.0

    camera = Camera(
        lookfrom=Vec3(478.0, 278.0, -600.0),
        lookat=Vec3(278.0, 278.0, 0.0),
        up=Vec3(0.0, 1.0, 0.0),
        aspect_ratio=aspect_ratio,
        vfov=40.0,
        defocus_angle=0.0,
        background=Vec3(0.0, 0.0, 0.0),
    )

    render(camera, world, aspect_ratio, width=image_width, samples_per_pixel=samples_per_pixel, max_depth=max_depth)


if __name__ == "__main__":
    match 10:
        case 1: bouncing_spheres()
        case 2: checkered_spheres()
        case 3: earth()
        case 4: perlin_spheres()
        case 5: quads()
        case 6: simple_light()
        case 7: cornell_box()
        case 8: cornell_smoke()
        case 9: final_scene(800, 10000, 40)
        case 10: final_scene(600, 2500, 40)
