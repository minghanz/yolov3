from torch.utils.data.sampler import Sampler
import random

class BatchFrozenSampler(Sampler):
    """This is a batch sampler. Each generated batch itself is fixed, but the sequence of the batches are random. 
    We need fixed batches because LoadImagesAndLabels() in datasets.py sorted the images according to aspect ratio and batch_shapes is already defined there. 
    We want to shuffle the sequence of the batches because otherwise,
    the images of different sources are likely to be fed into the network sequencially (i.e. all images from source A first, then all images from source B), which may not be good. 
    """

    def __init__(self, n_items, batch_size, drop_last):
        n_batches = n_items//batch_size if drop_last else (n_items+batch_size-1)//batch_size

        self.batched_idx = [None]*n_batches
        for i in range(n_batches):
            if i == n_batches-1 and not drop_last:
                self.batched_idx[i] = list(range(i*batch_size, n_items))
            else:
                self.batched_idx[i] = list(range(i*batch_size, (i+1)*batch_size))

    def __iter__(self):
        random.shuffle(self.batched_idx)
        return iter(self.batched_idx)

    def __len__(self):
        return len(self.batched_idx)