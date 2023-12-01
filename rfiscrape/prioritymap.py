"""A prioritised map like structure."""

import heapq
from collections.abc import Callable, Hashable
from typing import Any


class OutOfOrderError(RuntimeError):
    """Tried to insert entries out of order."""


class FullContainerError(RuntimeError):
    """The container is already full."""


PriorityType = int | float | str
KeyType = Hashable
ValueType = Any


class PriorityMap:
    """A key-value map with prioritised item and maximum capacity.

    Parameters
    ----------
    maxlength
        The maximum number of items in the map.
    strict
        If set, calls which try to add an item lower than the current lowest priority
        will raise an `OutOfOrder` exception. This ensures that the items returned
        from a series of `.pop()` calls would be in strict priority order.
    ignore_existing
        If the key currently exists silently return. If not set, a `KeyError` exception
        will be raised.
    """

    def __init__(
        self, maxlength: int, strict: bool = False, ignore_existing: bool = False
    ) -> None:
        self._maxlength = maxlength
        self._strict = strict
        self._ignore_existing = ignore_existing

        # Initialise the storage
        self._items = {}
        self._priorities = []

    def push(
        self,
        key: KeyType,
        value: ValueType | None = None,
        priority: PriorityType | None = None,
        call: Callable[[], ValueType] | None = None,
        _bump: bool = False,
    ) -> None:
        """Add an item to the map.

        Parameters
        ----------
        key
            For looking up the item.
        value
            The value for the item.
        priority
            An optional priority for the item. If not provided use the key itself.
        strict
            If set, calls which try to add an item lower than the current lowest will
            return `None`, rather than the item. This ensures that the items returned
            from a series of pop calls would be in strict priority order.
        call
            If set, value is assumed to be a callable and is evaluated only when the
            item is added. This saves creating expensive values if they would just
            result in a no-op.
        ignore_existing
            If the key currently exists silently return. If not set, an exception will
            be raised.
        """
        # _bump is an internal argument allowing us to temporarily exceed the size by 1.

        # Decide what to do if the key exists
        if key in self:
            if self._ignore_existing:
                return
            raise KeyError(f'Key "{key}" already exists in map.')

        if len(self._items) == (self._maxlength + (1 if _bump else 0)):
            raise FullContainerError(
                "Too many items in map. "
                "Currently {len(self._items)} but maxlength is {self._maxlength}.",
            )

        priority_pair = self._priority_pair(priority, key)

        # Don't do anything in strict mode if the new item would have too low priority
        if self._strict and self._priorities and priority_pair < self._priorities[0]:
            raise OutOfOrderError(
                f"Priority of new item {priority_pair[0]} is lower than "
                f"the lowest in the map {self._priorities[0]}.",
            )

        self._items[key] = call() if call else value
        heapq.heappush(self._priorities, priority_pair)

    def get(self, key: KeyType) -> ValueType:
        """Get the value corresponding to the key.

        Parameters
        ----------
        key
            The key to lookup.

        Returns
        -------
        value
            The corresponding value.
        """
        return self._items[key]

    __getitem__ = get

    def __len__(self) -> int:
        """Number of items in the map."""
        return len(self._items)

    def __contains__(self, key: KeyType) -> bool:
        """Determine if the item in the map already."""
        return key in self._items

    def pop(self) -> tuple[KeyType, ValueType]:
        """Remove the lowest priority item and its key-value pair.

        Returns
        -------
        key, value
            The key and value for the item.
        """
        if len(self) == 0:
            raise IndexError("pop called on empty PriorityMap.")

        _, key = heapq.heappop(self._priorities)
        return key, self._items.pop(key)

    def pushpop(
        self,
        key: Hashable,
        value: ValueType | None = None,
        priority: PriorityType | None = None,
        call: Callable[[], ValueType] | None = None,
    ) -> tuple[KeyType, ValueType] | tuple[None, None]:
        """Add an item to the map and return the lowest priority item only if it's full.

        This may return the item being passed if it is lower priority that the current
        lowest.

        Parameters
        ----------
        key
            For looking up the item.
        value
            The value for the item.
        priority
            An optional priority for the item. If not provided use the key itself.
        strict
            If set calls which try to add an item lower than the current lowest will
            return `None`, rather than the item. This ensures that the items returned
            from a series of calls are in strict priority order.
        call
            If set, value is assumed to be a callable and is evaluated only when the
            item is added. This saves creating expensive values if they would just
            result in a no-op.
        ignore_existing
            If the key currently exists, then do not replace it.

        Returns
        -------
        key_value
            Returns `None` if the map is not full, otherwise return the lowest key,
            value pair.
        """
        # We could do a slightly more efficient combined update, but this is much
        # simpler
        self.push(key, value=value, priority=priority, call=call, _bump=True)

        if len(self) > self._maxlength:
            return self.pop()
        return None, None

    def peek(self) -> tuple[PriorityType, KeyType]:
        """Return the lowest priority-key pair."""
        return self._priorities[0]

    def full(self) -> bool:
        """Check if the map is full."""
        return len(self) == self._maxlength

    def _priority_pair(
        self, priority: PriorityType | None, key: KeyType
    ) -> tuple[PriorityType | KeyType, KeyType]:
        """Create the priority, key pair."""
        if priority is None:
            return (key, key)
        return (priority, key)
