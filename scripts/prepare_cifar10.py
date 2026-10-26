from pathlib import Path
from torchvision.datasets import CIFAR10
root=Path(__file__).resolve().parents[1]/'data'
for train in (True,False):CIFAR10(root,train=train,download=True)
