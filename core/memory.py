"""
Memory management — circular buffers and object pools.
Prevents unbounded growth; explicit cleanup; no hidden allocations.
"""
from __future__ import annotations

import math
from typing import TypeVar, Generic, Optional, Callable, Iterator

T = TypeVar("T")


class CircularBuffer(Generic[T]):
    """
    Fixed-capacity circular buffer. Overwrites oldest entries when full.
    Used for rolling windows in indicators — no list growth, no slicing copies.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"CircularBuffer capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._data: list[Optional[T]] = [None] * capacity
        self._head = 0      # next write position
        self._count = 0

    def push(self, value: T) -> None:
        self._data[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    @property
    def full(self) -> bool:
        return self._count == self._capacity

    @property
    def count(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._capacity

    def newest(self) -> Optional[T]:
        if self._count == 0:
            return None
        idx = (self._head - 1) % self._capacity
        return self._data[idx]

    def oldest(self) -> Optional[T]:
        if self._count == 0:
            return None
        if self._count < self._capacity:
            return self._data[0]
        return self._data[self._head]

    def to_list(self) -> list[T]:
        """Return items oldest-first."""
        if self._count == 0:
            return []
        if self._count < self._capacity:
            return [v for v in self._data[:self._count] if v is not None]
        start = self._head
        result = []
        for i in range(self._capacity):
            idx = (start + i) % self._capacity
            v = self._data[idx]
            if v is not None:
                result.append(v)
        return result

    def __iter__(self) -> Iterator[T]:
        return iter(self.to_list())

    def __len__(self) -> int:
        return self._count

    def clear(self) -> None:
        self._data = [None] * self._capacity
        self._head = 0
        self._count = 0


class FloatCircularBuffer:
    """
    Optimised circular buffer for float arrays — avoids boxing overhead.
    Supports rolling sum and mean without recomputing from scratch.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._data = [0.0] * capacity
        self._head = 0
        self._count = 0
        self._sum = 0.0

    def push(self, value: float) -> None:
        if not math.isfinite(value):
            value = 0.0
        old = self._data[self._head]
        if self._count == self._capacity:
            self._sum -= old
        self._data[self._head] = value
        self._sum += value
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    @property
    def full(self) -> bool:
        return self._count == self._capacity

    @property
    def count(self) -> int:
        return self._count

    def mean(self) -> float:
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def total(self) -> float:
        return self._sum

    def newest(self) -> float:
        if self._count == 0:
            return 0.0
        return self._data[(self._head - 1) % self._capacity]

    def oldest(self) -> float:
        if self._count == 0:
            return 0.0
        if self._count < self._capacity:
            return self._data[0]
        return self._data[self._head]

    def to_list(self) -> list[float]:
        if self._count == 0:
            return []
        if self._count < self._capacity:
            return list(self._data[:self._count])
        start = self._head
        return [self._data[(start + i) % self._capacity] for i in range(self._capacity)]

    def clear(self) -> None:
        self._data = [0.0] * self._capacity
        self._head = 0
        self._count = 0
        self._sum = 0.0


class ObjectPool(Generic[T]):
    """
    Reusable object pool — avoids GC pressure from frequent allocations.
    Objects are reset via a provided reset_fn before reuse.
    """

    def __init__(self, factory: Callable[[], T], reset_fn: Callable[[T], None], max_size: int = 64) -> None:
        self._factory = factory
        self._reset_fn = reset_fn
        self._pool: list[T] = []
        self._max_size = max_size

    def acquire(self) -> T:
        if self._pool:
            obj = self._pool.pop()
            self._reset_fn(obj)
            return obj
        return self._factory()

    def release(self, obj: T) -> None:
        if len(self._pool) < self._max_size:
            self._pool.append(obj)

    def clear(self) -> None:
        self._pool.clear()

    @property
    def available(self) -> int:
        return len(self._pool)
