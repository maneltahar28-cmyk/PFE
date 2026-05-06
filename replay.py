import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, important_fraction=0.25):
        self.capacity = capacity
        self.main_capacity = int(capacity * (1.0 - important_fraction))
        self.important_capacity = max(1, capacity - self.main_capacity)

        self.main_buffer = deque(maxlen=self.main_capacity)
        self.important_buffer = deque(maxlen=self.important_capacity)

    def push(self, state, action, reward, next_state, done, *extra):
        transition = (state, action, reward, next_state, done, *extra)

        if abs(float(reward)) >= 2.5:
            self.important_buffer.append(transition)
        else:
            self.main_buffer.append(transition)

    def sample(self, batch_size):
        n_important = min(len(self.important_buffer), max(1, int(batch_size * 0.30)))
        n_main = batch_size - n_important

        if len(self.main_buffer) < n_main:
            n_important = min(batch_size - len(self.main_buffer), len(self.important_buffer))
            n_main = batch_size - n_important

        batch = []

        if n_main > 0 and len(self.main_buffer) >= n_main:
            batch.extend(random.sample(self.main_buffer, n_main))

        if n_important > 0 and len(self.important_buffer) >= n_important:
            batch.extend(random.sample(self.important_buffer, n_important))

        if len(batch) < batch_size:
            merged = list(self.main_buffer) + list(self.important_buffer)
            missing = batch_size - len(batch)
            if len(merged) >= missing:
                batch.extend(random.sample(merged, missing))

        random.shuffle(batch)
        return batch

    def __len__(self):
        return len(self.main_buffer) + len(self.important_buffer)