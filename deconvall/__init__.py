import importlib.metadata

#__version__ = importlib.metadata.version("deconvall")

from .approximation import Approximation
from .utils import coarse_grain_anndata
from .markers.markers import get_markers, get_all_markers
