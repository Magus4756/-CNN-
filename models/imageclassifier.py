# encode=utf-8

from torch.nn import *

from models.backbone import BackBone
from models.classifier import Classifier
from utils.model_init import initialize_weights


class ImageClassifier(Module):

    def __init__(self, config):
        super(ImageClassifier, self).__init__()
        self.cfg = config
        self.epoch = 0
        self.back_bone = _make_back_bone(self.cfg)
        self.classifier = _make_classifier(self.cfg)
        initialize_weights(self)
        # 其他参数
        self.epoch = 0

    def forward(self, x):
        x = self.back_bone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_model_state(self):
        L = []
        for param_tensor in self.state_dict():
            L.append('%s\t%s' % (param_tensor, self.state_dict()[param_tensor].size()))
        return L

    def get_optimizer_state(self):
        L = ['learning rate: %f' % self.optimizer.state_dict()['param_groups'][0]['lr'],
             'momentum: %f' % self.optimizer.state_dict()['param_groups'][0]['momentum'],
             'weight decay: %f' % self.optimizer.state_dict()['param_groups'][0]['weight_decay']]
        return L


def _make_back_bone(cfg):
    return BackBone(cfg)


def _make_classifier(cfg):
    # 新建全连接层
    return Classifier(cfg)


if __name__ == '__main__':
    from utils import load_config
    cfg = load_config('./config/VGG11.yaml')
    net = ImageClassifier(cfg)
    print(net)
