import numpy as np
from secml.array import CArray
from secml.data import CDataset
from secml.data.loader import CDataLoaderCIFAR100  # CIFAR-100 için loader

def identity(x):
    return x

def img_to_tensor(X):
    # Görüntü verilerini 0-1 aralığına normalize etme
    return X / 255.0

def split_train_val(tr_data, n_tr, n_val, transform=identity):
    # Eğitim ve doğrulama setlerini ayırma
    val = None
    tr = tr_data[:n_tr, :]
    tr.X = transform(tr.X)
    if n_val > 0:
        val = tr_data[n_tr:, :]
        val.X = transform(val.X)
    return tr, val

def filter_transform(ds, labels, n_ds=None, transform=img_to_tensor, bin_label=False):
    valid = [i for i, y in enumerate(ds.Y) if y in labels]
    if n_ds is not None:
        if len(valid) < n_ds:
            print(f"Warning: Requested sample size {n_ds} is larger than available {len(valid)}. Using all available samples.")
            n_ds = len(valid)
        valid = CArray(np.random.choice(a=valid, size=n_ds, replace=False))
    x = ds.X[valid, :]
    y = ds.Y[valid]
    if bin_label:
        y = y == labels[0]
    return CDataset(x=transform(x), y=y.astype(int))

def get_cifar_loader(n_tr, n_ts, labels, transform, bin_label=False):
    # CIFAR-100 verisini yükle
    loader = CDataLoaderCIFAR100()
    train, test = loader.load(val_size=0)

    data = []
    n_ds = [n_tr, n_ts]
    for i, ds in enumerate((train, test)):
        ds_f = filter_transform(
            ds, labels=labels, n_ds=n_ds[i], transform=transform, bin_label=bin_label
        )
        data.append(ds_f)
    return data

def load_data100(n_tr, n_ts, n_val=0, labels=None, transform=img_to_tensor):
    bin_l = False
    # Varsayılan olarak tüm CIFAR-100 sınıfları (0-99) seçilir
    if labels is None:
        labels = tuple(range(0, 100))
    # İkili sınıflandırma için iki etiket verilmişse bin_label True olur
    elif len(labels) == 2:
        bin_l = True
    train, test = get_cifar_loader(n_tr + n_val, n_ts, labels, transform, bin_label=bin_l)
    tr, val = split_train_val(train, n_tr=n_tr, n_val=n_val)
    return tr, val, test