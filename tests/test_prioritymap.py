import pytest

from rfiscrape import prioritymap


def test_push():
    """Test the behaviour on pushing."""
    p = prioritymap.PriorityMap(3)

    p.push("a", 5)
    p.push("b", 6)
    p.push("c", 7)

    assert p["a"] == 5
    assert p["b"] == 6
    assert p["c"] == 7

    assert len(p) == 3

    with pytest.raises(prioritymap.FullContainerError):
        p.push("d", 8)

    assert len(p) == 3

    k, v = p.pushpop("d", 8)
    assert k == "a"
    assert v == 5

    assert "a" not in p
    assert p["d"] == 8
    assert len(p) == 3


def test_pop():
    """Test that popping yields the expected elements."""
    p = prioritymap.PriorityMap(3)

    p.push("c", 7)
    p.push("a", 5)
    p.push("b", 6)

    assert len(p) == 3
    assert p.pop() == ("a", 5)
    assert len(p) == 2
    assert p.pop() == ("b", 6)
    assert len(p) == 1
    assert p.pop() == ("c", 7)
    assert len(p) == 0

    with pytest.raises(IndexError):
        p.pop()


def test_strict():
    """Test the behaviour with strict ordering."""
    p = prioritymap.PriorityMap(3, strict=True)

    p.push("c", 7)
    assert len(p) == 1

    with pytest.raises(prioritymap.OutOfOrderError):
        p.push("a", 5)

    assert len(p) == 1
    p.push("e", 9)
    assert len(p) == 2
    p.push("d", 8)
    assert len(p) == 3

    p.pop()
    assert len(p) == 2
    with pytest.raises(prioritymap.OutOfOrderError):
        p.push("c", 7)


def test_existing():
    """Test the behaviour when trying to reinsert an existing key."""
    p = prioritymap.PriorityMap(3, ignore_existing=False)

    p.push("a", 5)
    p.push("b", 6)
    p.push("c", 7)

    with pytest.raises(KeyError):
        p.push("b", 17)

    with pytest.raises(KeyError):
        p.push("b", 17, priority="z")

    p = prioritymap.PriorityMap(3, ignore_existing=True)

    p.push("a", 5)
    p.push("b", 6)
    p.push("c", 7)

    p.push("b", 17)
    assert p["b"] == 6

    p.push("b", 17, priority="z")
    assert p["b"] == 6
