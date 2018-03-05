from .sampler import SequentialSampler, RandomSampler, BatchSampler
import collections
import numpy as np

def Stack(arrays):
    new_dim_array = [np.expand_dims(array, axis=0) for array in arrays]
    return np.concatenate(new_dim_array, axis=0)

def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if isinstance(batch[0], (np.ndarray, np.generic)):
        return Stack(batch)
    elif isinstance(batch[0], int):
        return np.array(batch, dtype=np.int)
    elif isinstance(batch[0], float):
        return np.array(batch, dtype=np.float)
    elif isinstance(batch[0], (str, bytes)):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    raise TypeError((error_msg.format(type(batch[0]))))

class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.sample_iter = iter(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self.sample_iter)  # may raise StopIteration
        batch = self.collate_fn([self.dataset[i] for i in indices])
        return batch

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 collate_fn=default_collate, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last

        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        self.sampler = sampler
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        
    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)
    