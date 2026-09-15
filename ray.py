from vec3 import Vec3
    
class Ray:
    def __init__(self, origin: Vec3, direction: Vec3, time: float = 0.0):
        self.origin = origin
        self.direction = direction
        self.time = time

    def at(self, t: float) -> Vec3:
        return self.origin + t * self.direction