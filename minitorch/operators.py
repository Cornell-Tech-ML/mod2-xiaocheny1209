"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply 'x' by 'y'"""
    return x * y


def id(x: float) -> float:
    """Return 'x' unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add 'x' and 'y'"""
    return x + y


def neg(x: float) -> float:
    """Negate 'x'"""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if 'x' is less than 'y'"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if 'x' is equal to 'y'"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of 'x' and 'y'"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if 'x' and 'y' are close, f(x) = |x - y| < 1e-2"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm of a number."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function of a number."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal (inverse) of a number."""
    if x == 0:
        raise ZeroDivisionError("Cannot calculate reciprocal of zero.")
    return 1.0 / x


def log_back(x: float, grad: float) -> float:
    """Computes the derivative of the natural logarithm function times a second argument."""
    if x <= 0:
        raise ValueError("Input must be positive for logarithm.")
    return grad / (x + EPS)


def inv_back(x: float, grad: float) -> float:
    """Computes the derivative of the reciprocal function times a second argument."""
    if x == 0:
        raise ZeroDivisionError("Cannot calculate derivative of reciprocal of zero.")
    # return -grad / (x**2)
    return -(1.0 / x**2) * grad


def relu_back(x: float, grad: float) -> float:
    """Computes the derivative of the ReLU function times a second argument."""
    return grad if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(f: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Applies a given function to each element of an iterable"""
    return [f(e) for e in ls]


def zipWith(
    f: Callable[[float, float], float], l1: Iterable[float], l2: Iterable[float]
) -> Iterable[float]:
    """Combines elements from two iterables using a given function"""
    return (f(x, y) for x, y in zip(l1, l2))


def reduce(f: Callable[[float, float], float], ls: Iterable[float]) -> float:
    """Reduces an iterable to a single value using a given function"""
    it = iter(ls)  # Convert the iterable to an iterator
    try:
        result = next(it)
    except StopIteration:
        return 0
        # raise ValueError(
        #     "Reduce operation requires at least one element in the iterable"
        # )
    for element in it:
        result = f(result, element)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    return map(neg, ls)


def addLists(l1: Iterable[float], l2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add, l1, l2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, ls)
