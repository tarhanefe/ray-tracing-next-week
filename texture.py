from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import floor, sin
from hitable import Interval
from vec3 import Vec3
from rtw_image import RTWImage
from perlin import Perlin

class Texture(ABC):
    @abstractmethod
    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        pass

@dataclass
class SolidColor(Texture):
    albedo: Vec3

    @classmethod
    def from_rgb(cls, red: float, green: float, blue: float) -> "SolidColor":
        return cls(Vec3(red, green, blue))

    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        return self.albedo

class CheckerTexture(Texture):
    def __init__(self, scale: float, even: Texture, odd: Texture):
        self.inv_scale = 1.0 / scale
        self.even = even
        self.odd = odd

    @classmethod
    def from_colors(cls, scale: float, c1: Vec3, c2: Vec3) -> "CheckerTexture":
        return cls(scale, SolidColor(c1), SolidColor(c2))

    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        # Which cube of side `scale` the point lies in, counted along each axis.
        x_integer = floor(self.inv_scale * p.x)
        y_integer = floor(self.inv_scale * p.y)
        z_integer = floor(self.inv_scale * p.z)

        is_even = (x_integer + y_integer + z_integer) % 2 == 0

        return self.even.value(u, v, p) if is_even else self.odd.value(u, v, p)

class ImageTexture(Texture):
    def __init__(self, image: "RTWImage"):
        self.image = image

    @classmethod
    def from_file(cls, filename: str) -> "ImageTexture":
        return cls(RTWImage(filename))

    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        if self.image.width() == 0 or self.image.height() == 0:
            return Vec3(0.0, 1.0, 1.0)  # Return cyan for missing image

        # Clamp UV coordinates to [0,1]
        u = Interval(0, 1).clamp(u)
        v = 1.0 - Interval(0, 1).clamp(v)


        # Convert to pixel coordinates
        i = int(u * self.image.width())
        j = int(v * self.image.height())
        pixel = self.image.pixel_data(i, j)
        color_scale = 1/255.0
        return Vec3(pixel[0] * color_scale, pixel[1] * color_scale, pixel[2] * color_scale)

class NoiseTexture(Texture):
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.perlin = Perlin()

    def value(self, u: float, v: float, p: Vec3) -> Vec3:
        noise_value = self.perlin.noise(self.scale * p)
        return Vec3(0.5,0.5,0.5) * (1 + sin(self.scale * p.z + 10 * self.perlin.turbulance(p,7)))