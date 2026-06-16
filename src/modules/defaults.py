from torch import nn
#from bcos.models.resnet import LogitLayer, BasicBlock, BcosConv2d
from src.modules.bcos import BcosConv2d_unnormed
from src.modules.bcos import BcosLinear
from src.modules import norms

DEFAULT_NORM_LAYER = norms.NoBias(norms.GNLayerNormUncentered2d)
# DEFAULT_NORM_LAYER = (norms.DetachableLayerNorm)
#DEFAULT_NORM_LAYER = torch.nn.Identity

DEFAULT_ACT_LAYER = nn.ReLU

DEFAULT_CONV_LAYER = BcosConv2d_unnormed
#DEFAULT_CONV_LAYER = BcosConv2d_unnormed
DEFAULT_CONV_LAYER_ADAPTER = BcosConv2d_unnormed
DEFAULT_LINEAR_LAYER = BcosLinear