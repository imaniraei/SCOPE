"""Model components for SCOPE."""
from .se import SqueezeExcitation
from .cbam import CBAM, ChannelAttention, SpatialAttention
from .mbconv import MBConvCBAM
from .efficientnet import EfficientNetCBAMBackbone
from .adaptive_bifpn import AdaptiveBiFPN
from .heads import ClassificationHead, BoxRegressionHead
from .scope import ScopeDetector, ScopeScaleConfig, SCOPE_SCALES, build_scope
