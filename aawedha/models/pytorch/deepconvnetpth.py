from aawedha.models.pytorch.torch_utils import LineardWithConstraint
from aawedha.models.pytorch.torch_utils import Conv2dWithConstraint
from aawedha.models.pytorch.torchmodel import TorchModel
from torch.nn.functional import elu
from torch import flatten
from torch import nn


# based on the Keras implementation by the EEGnet authors
class DeepConvNetPTH(TorchModel):
    """ Pytorch implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """
    def __init__(self, nb_classes, Chans=64, Samples=256,
                dropoutRate=0.5, device="cuda", name="DeepConvNetPTH"):
        super().__init__(device, name)
        division_rate = ((Samples - 16) // 8) - 1
        # block1
        self.conv = Conv2dWithConstraint(1, 25, (1,5), bias=False, max_norm=2., axis=(0,1,2))
        self.conv1 = Conv2dWithConstraint(25, 25, (Chans, 1),  bias=False, max_norm=2., axis=(0,1,2))
        self.bn1 = nn.BatchNorm2d(25, momentum=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.do1 = nn.Dropout(p=dropoutRate) 
        # block2
        self.conv2 = Conv2dWithConstraint(25, 50, (1,5), bias=False, max_norm=2., axis=(0,1,2))
        self.bn2 = nn.BatchNorm2d(50, momentum=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.do2 = nn.Dropout(p=dropoutRate) 
        # block3
        self.conv3 = Conv2dWithConstraint(50, 100, (1,5), bias=False, max_norm=2., axis=(0,1,2))
        self.bn3 = nn.BatchNorm2d(100, momentum=0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.do3 = nn.Dropout(p=dropoutRate) 
        # block4
        self.conv4 = Conv2dWithConstraint(100, 200, (1,5), bias=False, max_norm=2., axis=(0,1,2))
        self.bn4 = nn.BatchNorm2d(200, momentum=0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.do4 = nn.Dropout(p=dropoutRate) 
        #
        self.dense = LineardWithConstraint(200 * (Samples // division_rate), nb_classes, max_norm=0.5)

        self.initialize_glorot_uniform()
        

    def forward(self, x):        
        x = self._reshape_input(x)
        x = self.do1(self.pool1(elu(self.bn1(self.conv1(self.conv(x))))))
        x = self.do2(self.pool2(elu(self.bn2(self.conv2(x)))))
        x = self.do3(self.pool3(elu(self.bn3(self.conv3(x)))))
        x = self.do4(self.pool4(elu(self.bn4(self.conv4(x)))))
        x = flatten(x, 1)
        x = self.dense(x)
        return x



 