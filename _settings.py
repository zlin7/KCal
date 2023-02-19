import os
import getpass
import sys

__USERNAME = getpass.getuser()
DATA_PATH = "/srv/local/data"
WORKSPACE = os.path.join(f'/srv/local/data/{__USERNAME}/KCal/', "Temp")
if not os.path.isdir(WORKSPACE): os.makedirs(WORKSPACE)

_ON_SERVER = True

__CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

#==============================Data Related
ISRUC_NAME = 'ISRUC_SLEEP1'
CIFAR10_NAME = 'CIFAR10'
CIFAR100_NAME = 'CIFAR100'
IIICSup_NAME = "IIICSup_1109"
ImageNet1K_NAME = "ImageNet1K"
SVHN_NAME = 'SVHN'
ECG_NAME = 'ECG'

ISRUC_PATH = os.path.join(DATA_PATH, ISRUC_NAME)
CIFAR10_PATH = os.path.join(DATA_PATH, CIFAR10_NAME)
CIFAR100_PATH = os.path.join(DATA_PATH, CIFAR100_NAME)
IIICSup_PATH = os.path.join(DATA_PATH, 'IIIC_data')
ImageNet1K_PATH = os.path.join(DATA_PATH, ImageNet1K_NAME)
SVHN_PATH = os.path.join(DATA_PATH, SVHN_NAME)
ECG_PATH = os.path.join(DATA_PATH, ECG_NAME)

LOG_OUTPUT_DIR = os.path.join(WORKSPACE, 'logs')
RANDOM_SEED = 7


#Trained DNN keys
_TRAINED_KEYS = {
    (CIFAR10_NAME, 'ViT'): 'ViTB16_timm-20211226_013412',
                 (CIFAR10_NAME, 'Mixer'): 'MixerB16_timm-20220102_222852',
                 (CIFAR100_NAME, 'ViT'): 'ViTB16_timm-20211215_013918',
                 (CIFAR100_NAME, 'Mixer'): 'MixerB16_timm-20220102_191050',
                 (SVHN_NAME, 'ViT'): 'ViTB16_timm-20211226_235923',
                 (SVHN_NAME, 'Mixer'): 'MixerB16_timm-20220103_085510',

                 (ImageNet1K_NAME, 'InceptResNet'): 'inception_resnet_v2',

                 (IIICSup_NAME, 'ResNet'): 'CNNEncoder2D_IIIC-20211226_131303',
                 (ECG_NAME, 'MINA'): 'FreqNet-20220112_1729130.01_plat',
                 (ISRUC_NAME, 'ResNet'): 'CNNEncoder2D_ISRUC-20220102_225606',
                 }


_KERNEL_KEYS ={(IIICSup_NAME, 'ResNet'): 'ProjectionTrainer-20220109_195148Skip-ELU',
                (ECG_NAME, 'MINA'): 'ProjectionTrainer-20220113_024806Skip-ELU8',
                (ISRUC_NAME, 'ResNet'): 'ProjectionTrainer-20220109_182951Skip-ELU', 
    (CIFAR10_NAME, 'Mixer'): "ProjectionTrainer-20220109_203252Skip-ELU",
    (CIFAR100_NAME, 'Mixer'): "ProjectionTrainer-20220109_171955Skip-ELU",
    (SVHN_NAME, 'Mixer'): "ProjectionTrainer-20220109_203252Skip-ELU",
    (CIFAR10_NAME, 'ViT'): "ProjectionTrainer-20220109_173301Skip-ELU",
    (CIFAR100_NAME, 'ViT'): "ProjectionTrainer-20220109_173301Skip-ELU",
    (SVHN_NAME, 'ViT'): "ProjectionTrainer-20220109_214920Skip-ELU",
}


NCOLS = 80

NEPTUNE_PROJECT = ''
NEPTUNE_API_TOKEN = ""