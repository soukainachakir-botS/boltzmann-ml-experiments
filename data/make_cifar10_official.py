"""Builds data/cifar10_official.npz from the official CIFAR-10 files (python version).
Download cifar-10-python.tar.gz from https://www.cs.toronto.edu/~kriz/cifar.html and extract it into data/,
so that data/cifar-10-batches-py/ exists. Then run from the repository root:  python data/make_cifar10_official.py
The MD5 checksums of the official files are verified before they are read.
Each image is a row of 3,072 values (1,024 red, then 1,024 green, then 1,024 blue), as in the official files.
"""
import hashlib, pickle, numpy as np
D='data/cifar-10-batches-py/'
MD5={'data_batch_1':'c99cafc152244af753f735de768cd75f','data_batch_2':'d4bba439e000b95fd0a9bffe97cbabec',
     'data_batch_3':'54ebc095f3ab1f0389bbae665268c751','data_batch_4':'634d18415352ddfa80567beed471001a',
     'data_batch_5':'482c414d41f54cd18b22e5b47cb7c3cb','test_batch':'40351d587109b95175f43aff81a1287e'}
def read(f):
    raw=open(D+f,'rb').read()
    assert hashlib.md5(raw).hexdigest()==MD5[f], f'checksum mismatch for {f}: not the official file'
    d=pickle.loads(raw,encoding='bytes')
    return np.asarray(d[b'data'],dtype=np.uint8), np.asarray(d[b'labels'])
tr=[read(f'data_batch_{i}') for i in range(1,6)]
Xtr=np.vstack([a for a,_ in tr]); ytr=np.concatenate([b for _,b in tr]); Xte,yte=read('test_batch')
assert Xtr.shape==(50000,3072) and Xte.shape==(10000,3072)
np.savez('data/cifar10_official.npz',Xtr=Xtr,ytr=ytr,Xte=Xte,yte=yte)
print(Xtr.shape,Xte.shape,np.bincount(ytr),np.bincount(yte))
