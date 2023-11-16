import heapq
from typing import Callable, Hashable, Any


class OutOfOrder(RuntimeError):
    """Tried to insert entries out of order."""


class FullContainer(RuntimeError):
    """The container is already full."""


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
        self, maxlength: int, strict: bool = False, ignore_existing: bool = False,
    ) -> None:
        self._maxlength = maxlength
        self._strict = strict
        self._ignore_existing = ignore_existing

        # Initialise the storage
        self._items = {}
        self._priorities = []

    def push(
        self,
        key: Hashable,
        value: Any | None = None,
        priority: Any | None = None,
        call: Callable[[], Any] | None = None,
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
        # _bump is an internal argument that allows us to temporarily exceed the size by 1.

        # Decide what to do if the key exists
        if key in self:
            if self._ignore_existing:
                return None
            else:
                raise KeyError(f'Key "{key}" already exists in map.')

        if len(self._items) == (self._maxlength + (1 if _bump else 0)):
            raise FullContainer(
                "Too many items in map. "
                "Currently {len(self._items)} but maxlength is {self._maxlength}."
            )

        priority_pair = self._priority_pair(priority, key)

        # Don't do anything in strict mode if the new item would have too low priority
        if self._strict and self._priorities and priority_pair < self._priorities[0]:
            raise OutOfOrder(
                f"Priority of new item {priority_pair[0]} is lower than "
                f"the lowest in the map {self._priorities[0]}."
            )

        self._items[key] = call() if call else value
        heapq.heappush(self._priorities, priority_pair)

    def get(self, key: Any) -> Any:
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

    def __contains__(self, key: Hashable) -> bool:
        """Is the item in the map already?"""
        return key in self._items

    def pop(self) -> tuple[Hashable, Any]:
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
        value: Any | None = None,
        priority: Any | None = None,
        call: Callable[[], Any] | None = None,
    ) -> tuple[Hashable, Any] | tuple[None, None]:
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
        self.push(
            key,
            value=value,
            priority=priority,
            call=call,
            _bump=True,
        )

        if len(self) > self._maxlength:
            return self.pop()
        else:
            return None, None

        # # If we are not full, we can push and return None
        # if len(self) < self._maxlength:
        #     self.push(key, value, priority=priority, call=call)
        #     return None

        # # If the priority is lower than the current lowest decide whether to return the
        # # item, or silently return None
        # pair = self._priority_pair(priority, key)
        # if pair < self._priorities[0]:
        #     return None if strict else (key, value)

        # # Otherwise we actually need to do something. push-pop the key, insert the new
        # # item, and then return the old item
        # _, oldkey = heapq.heappushpop(self._priorities, pair)
        # self._items[key] = value() if call else value

        # return oldkey, self._items.pop(oldkey)

    def peek(self) -> tuple[Any, Hashable]:
        """Return the lowest priority-key pair."""
        return self._priorities[0]

    def full(self):
        """Is the map full?"""
        return len(self) == self._maxlength

    def _priority_pair(self, priority, key):
        """Create the priority, key pair."""
        if priority is None:
            return (key, key)
        else:
            return (priority, key)
