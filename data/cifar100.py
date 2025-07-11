import pickle

with open('/Users/nilufergulciftci/Desktop/beta_poisoning-defenses/data/cifar-100-python/meta', 'rb') as f:
    meta_data = pickle.load(f)

print(meta_data['fine_label_names'])