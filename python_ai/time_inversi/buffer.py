from collections import deque
from .selfplay import Sample


class ReplayBuffer:
    def __init__(self, capacity=50000):
        if capacity < 1:
            raise ValueError('Buffer capacity must be positive')
        self.samples = deque(maxlen=capacity)

    def __len__(self):
        return len(self.samples)

    def add(self, samples):
        self.samples.extend(samples)

    def sample(self, size, rng):
        if not 0 < size <= len(self):
            raise ValueError('Not enough samples for this batch')
        data = list(self.samples)
        return [data[int(i)] for i in rng.choice(len(data), size=size, replace=False)]

    def to_dict(self):
        return dict(capacity=self.samples.maxlen, samples=[s.to_dict() for s in self.samples])

    @classmethod
    def from_dict(cls, data):
        buffer = cls(data['capacity'])
        buffer.add(Sample.from_dict(s) for s in data['samples'])
        return buffer
