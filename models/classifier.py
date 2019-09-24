import copy

from torch.nn import *

from utils.model_init import initialize_weights


class Classifier(Module):

    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.drop = Dropout(0.5)
        self.FCN = _make_FCN(self)
        initialize_weights(self)

    def forward(self, x):
        x = self.drop(x)
        x = self.FCN(x)
        return x


def _make_FCN(model):
    in_channel = model.cfg.MODEL.BACKBONE.OUT_CHENNELS
    out_channel = model.cfg.MODEL.CLASSIFIER.NUM_CLASSES
    layer = Linear(in_channel, out_channel)
    return layer
