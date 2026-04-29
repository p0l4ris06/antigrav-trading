import multiprocessing
import threading
import numpy as np
from typing import List

class PageHinkleyDrift:
    def __init__(self, delta=0.005, threshold=0.05, alpha=0.99):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mean = 0.0
        self.sum = 0.0
        self.n = 0
        self.m_t = 0.0
        self.M_t = 0.0

    def update(self, x):
        self.n += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x
        self.m_t += (x - self.mean - self.delta)
        self.M_t = max(self.M_t, self.m_t)
        
        drift = self.M_t - self.m_t
        if drift > self.threshold:
            self._reset()
            return True
        return False

    def _reset(self):
        self.mean = 0.0; self.sum = 0.0; self.n = 0; self.m_t = 0.0; self.M_t = 0.0

def shadow_train_task(shared_weights, lock, shape):
    new_weights = np.random.randn(*shape).astype(np.float32)
    with lock:
        np.copyto(shared_weights, new_weights)

class Daemon:
    def __init__(self, model):
        self.ph_test = PageHinkleyDrift()
        self.model = model
        self.live_weights_shape = model.policy.action_net.weight.detach().numpy().shape
        self.shared_weights = multiprocessing.Array('f', int(np.prod(self.live_weights_shape)))
        self.lock = multiprocessing.Lock()

    def monitor(self, rolling_sharpe):
        if self.ph_test.update(rolling_sharpe):
            p = multiprocessing.Process(target=shadow_train_task, args=(self.shared_weights, self.lock, self.live_weights_shape))
            p.start()
            
    def apply_updates(self):
        with self.lock:
            pass # State dict load stub
