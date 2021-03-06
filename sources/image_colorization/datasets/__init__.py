from .utils import view_images
from .utils import figure_to_image

from .quantized_colors import read_image
from .quantized_colors import ColorizedSoftEncoding
from .quantized_colors import decode_soft_encoding_list_rgb
from .quantized_colors import decode_soft_encoding_batch_rgb
from .quantized_colors import decode_soft_encoding_image_rgb
from .quantized_colors import decode_soft_encoding_batch_ab

from .normal_colors_v1 import decode_regression_list_image
from .normal_colors_v1 import decode_regression_image
from .normal_colors_v1 import decode_regression_batch_image