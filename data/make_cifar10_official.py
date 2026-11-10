"""Builds data/cifar10_official.npz (official 50,000/10,000 partition) from the image mirror.
Run from the repository root after: git clone https://github.com/YoongiKim/CIFAR-10-images data/CIFAR-10-images
"""
import os, numpy as np
from PIL import Image
def build(root):
    classes=sorted(os.listdir(root)); X=[];y=[]
    for k,c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(root,c))):
            X.append(np.asarray(Image.open(os.path.join(root,c,f)).convert('RGB')).reshape(-1)); y.append(k)
    return np.array(X,dtype=np.uint8),np.array(y),classes
Xtr,ytr,c1=build('data/CIFAR-10-images/train'); Xte,yte,c2=build('data/CIFAR-10-images/test'); assert c1==c2
np.savez('data/cifar10_official.npz',Xtr=Xtr,ytr=ytr,Xte=Xte,yte=yte)
print(Xtr.shape,Xte.shape,np.bincount(ytr),np.bincount(yte))
