import torch

def build_data_loader(dataset, consumed_samples, global_batch_size):
    if dataset is None:
        return None
    # batch sampler
    batch_sampler = LLaMANormalSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        global_batch_size=global_batch_size
    )
    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

# directly return the whole global batch, will be split into chunks later
class LLaMANormalSampler:

    def __init__(self, total_samples, consumed_samples, 
                 global_batch_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.global_batch_size = global_batch_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.global_batch_size:
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch