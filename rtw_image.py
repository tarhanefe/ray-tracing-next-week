import os
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError

class RTWImage:
    BYTES_PER_PIXEL = 3
    # stb_image's stbi_loadf converts 8-bit images to linear floats with this gamma; Pillow
    # gives the raw bytes, so load() applies it to match.
    STB_LDR_TO_HDR_GAMMA = 2.2
    # Returned by pixel_data() when there is no image data.
    MAGENTA = (255, 0, 255)

    def __init__(self, image_filename: str | None = None):
        # Loads image data from the specified file. If the RTW_IMAGES environment variable is
        # defined, looks only in that directory for the image file. If the image was not found,
        # searches for the specified image file first from the current directory, then in the
        # images/ subdirectory, then the _parent's_ images/ subdirectory, and then _that_
        # parent, on so on, for six levels up. If the image was not loaded successfully,
        # width() and height() will return 0.
        self.fdata: np.ndarray | None = None  # Linear floating point pixel data
        self.bdata: bytes | None = None       # Linear 8-bit pixel data
        self.image_width = 0                  # Loaded image width
        self.image_height = 0                 # Loaded image height
        self.bytes_per_scanline = 0

        if image_filename is None:
            return

        imagedir = os.environ.get("RTW_IMAGES")

        # Hunt for the image file in some likely locations.
        if imagedir and self.load(imagedir + "/" + image_filename):
            return
        if self.load(image_filename):
            return
        for levels_up in range(7):
            if self.load("../" * levels_up + "images/" + image_filename):
                return

        print(f"ERROR: Could not load image file '{image_filename}'.", file=sys.stderr)

    def load(self, filename: str) -> bool:
        # Loads the linear (gamma=1) image data from the given file name. Returns true if the
        # load succeeded. The resulting data buffer contains the three [0.0, 1.0]
        # floating-point values for the first pixel (red, then green, then blue). Pixels are
        # contiguous, going left to right for the width of the image, followed by the next row
        # below, for the full height of the image.
        try:
            with Image.open(filename) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        except (OSError, UnidentifiedImageError):
            # OSError covers a missing or unreadable file.
            return False

        self.image_height, self.image_width = rgb.shape[:2]
        self.fdata = (rgb / 255.0) ** self.STB_LDR_TO_HDR_GAMMA
        self.bytes_per_scanline = self.image_width * self.BYTES_PER_PIXEL
        self.convert_to_bytes()
        return True

    def width(self) -> int:
        return 0 if self.fdata is None else self.image_width

    def height(self) -> int:
        return 0 if self.fdata is None else self.image_height

    def pixel_data(self, x: int, y: int) -> tuple[int, int, int]:
        # Return the three RGB bytes of the pixel at x,y. If there is no image data, returns
        # magenta. (C++ returns a pointer into the buffer; a tuple is the Python equivalent.)
        if self.bdata is None:
            return self.MAGENTA

        x = self.clamp(x, 0, self.image_width)
        y = self.clamp(y, 0, self.image_height)

        index = y * self.bytes_per_scanline + x * self.BYTES_PER_PIXEL
        return self.bdata[index], self.bdata[index + 1], self.bdata[index + 2]

    @staticmethod
    def clamp(x: int, low: int, high: int) -> int:
        # Return the value clamped to the range [low, high).
        if x < low:
            return low
        if x < high:
            return x
        return high - 1

    @staticmethod
    def float_to_byte(value: float) -> int:
        if value <= 0.0:
            return 0
        if 1.0 <= value:
            return 255
        return int(256.0 * value)

    def convert_to_bytes(self):
        # Convert the linear floating point pixel data to bytes, storing the resulting byte
        # data in the `bdata` member.
        #
        # Same mapping as float_to_byte(), done on the whole array at once instead of a
        # per-component Python loop. Stored as `bytes` because indexing it yields plain ints,
        # which is much faster than indexing a numpy array once per ray.
        scaled = np.floor(256.0 * self.fdata)
        self.bdata = np.clip(scaled, 0, 255).astype(np.uint8).tobytes()
