import os
import numpy as np
import torch
import torch.utils.data
import random

# This is task specific
EXTENSIONS = ['.wav']
def load_sample(fname, normalize=True):
    from scipy.io.wavfile import read
    mat = read(fname)[1]
    mat = np.float32(mat)
    data = mat.squeeze()[None]
    if normalize:
        data = (data - data.mean()) / data.std()
        return data

def is_correct_extension(filename):
    """Checks if a file is a required sample.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known sample extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    samples = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_correct_extension(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    samples.append(item)

    return samples



def load_composed(fname, class_idx, transformations=[], **kwargs):
    data = load_sample(fname)
    for trans in transformations:
        data = trans(data)

    if len(data.shape) == 1:
        data = data[None, :]
    res = {
        'x': torch.from_numpy(data),
        #'features': torch.from_numpy(data[None, :]),
        'y': class_idx}
    return res


def batchify(batch):
    max_len = max(s['x'].size(-1) for s in batch)
    num_channels = batch[0]['x'].size(0)
    x_batch = torch.zeros(len(batch), num_channels, max_len)
    for idx in range(len(batch)):
        x_batch[idx, :, :batch[idx]['x'].size(-1)] = batch[idx]['x']

    y_batch = torch.LongTensor([s['y'] for s in batch])
    #feature_batch = torch.cat([s['features'] for s in batch], dim=0)


    res = {'x': x_batch,
           'y': y_batch
          }
    return res



def load_forked(fname, class_idx, transform=[], **kwargs):
    data = load_sample(fname)

    for trans in transform:
        data = trans(data)
    res = {}
    for forkname, fork_data in data.items():
        assert fork_data.shape, len(fork_data.shape) < 3
        if len(fork_data.shape) == 1:
            fork_data = fork_data[None, :]
        res[forkname] = {
            'x':th.from_numpy(fork_data),
            'y': class_idx
        }

    return res

def batchify_forked(batch):
    forked_res = {}
    for key in batch[0].keys():
        forked_res[key] = batchify(list(sample[key] for sample in batch))

    res = {'x': {}}
    for key, val in forked_res.items():
        res['x'][key] = val['x']
    # Every forks `y` is the same (at least should be)
    res['y'] = val['y']
    return res

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=[], forked=False, **kwargs):
        super(Dataset, self).__init__()
        self.root = root
        self.classes, self.class_idx = find_classes(root)
        self.samples = make_dataset(root, self.class_idx)
        self.num_classes = len(self.classes)
        self.transform = transform
        self.forked = forked
        self.loader = load_composed
        if forked:
            self.loader = load_forked


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        path, class_idx = self.samples[idx]
        return self.loader(path, class_idx, self.transform)

    def disjunct_split(self, ratio=.8):
        # Split keeps the ratio of classes
        A_samples = random.sample(self.samples, int(len(self.samples) * ratio))
        B_samples = set(self.samples) - set(A_samples)

        A = Dataset(
            root=self.root,
            transform=self.transform,
            forked=self.forked)
        A.samples = list(A_samples)

        B = Dataset(
            root=self.root,
            transform=self.transform,
            forked=self.forked)
        B.samples = list(B_samples)

        return A, B

    def save(self, fname):
        with open(fname, 'w') as f:
            f.writelines("%s,%d\n" % (l,idx) for l, idx in self.samples)

    def load(self, fname):
        with open(fname, 'r') as f:
            samples = [(l[:l.find(',')], int(l[l.find(',')+1:])) for l in f.readlines()]
        self.samples = samples




########################
# WRITE UNIT TEST HERE #
########################

if __name__ == '__main__':
    random.seed(42)
    dataset = Dataset('data/')
    assert(len(dataset) == 17636)
    train_set, eval_set = dataset.disjunct_split(.8)
    assert([len(train_set), len(eval_set)] == [14108, 3528])
    assert(len(set(train_set.samples).intersection(set(eval_set.samples))) == 0)

    train_producer = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=12, shuffle=True,
        num_workers=1, collate_fn=batchify)

    test_producer = torch.utils.data.DataLoader(
        dataset=eval_set, batch_size=4,
        num_workers=1, collate_fn=batchify)
