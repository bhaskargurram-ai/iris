# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Fast hashing of numpy and dask arrays, for comparing array values."""

import dask.array as da
import numpy as np
from xxhash import xxh3_64


def _normalise_floats(a: np.ndarray) -> np.ndarray:
    """Give floating point values that compare equal the same bytes.

    This applies to ``-0.0`` and ``0.0``, and to NaNs with different bit
    patterns, which are all treated as equal by
    :func:`iris.util.array_equal` with ``withnans=True``.

    Parameters
    ----------
    a :
        The array to normalise.

    Returns
    -------
    numpy.ndarray :
        The input array, or a normalised copy if it has a floating point dtype.

    """
    if a.dtype.kind == "f":
        a = np.array(np.ma.getdata(a))
        # Adding zero turns -0.0 into 0.0.
        a += 0
        a[np.isnan(a)] = np.nan
    return a


def _hash_ndarray(a: np.ndarray) -> np.ndarray:
    """Compute a hash from a numpy array.

    Calculates a 64-bit non-cryptographic hash of the provided array, using
    the fast ``xxhash`` hashing algorithm.

    Parameters
    ----------
    a :
        The array to hash.

    Returns
    -------
    numpy.ndarray :
        An array of shape (1,) containing the hash value.

    """
    # Include the array dtype as it is not preserved by `ndarray.tobytes()`.
    hash = xxh3_64(f"dtype={a.dtype}".encode("utf-8"))

    # Hash the bytes representing the array data.
    hash.update(b"data=")
    if np.ma.is_masked(a):
        # Hash only the unmasked data
        hash.update(_normalise_floats(a.compressed()).tobytes())
        # Hash the mask
        hash.update(b"mask=")
        hash.update(a.mask.tobytes())
    else:
        hash.update(_normalise_floats(a).tobytes())
    return np.frombuffer(hash.digest(), dtype=np.int64)


def _hash_chunk(
    x_chunk: np.ndarray,
    axis: tuple[int] | None,
    keepdims: bool,
) -> np.ndarray:
    """Compute a hash from a numpy array.

    This function can be applied to each chunk or intermediate chunk in
    :func:`~dask.array.reduction`. It preserves the number of input dimensions
    to facilitate combining intermediate results into intermediate chunks.

    Parameters
    ----------
    x_chunk :
        The array to hash.
    axis :
        Unused but required by :func:`~dask.array.reduction`.
    keepdims :
        Unused but required by :func:`~dask.array.reduction`.

    Returns
    -------
    numpy.ndarray :
        An array containing the hash value.

    """
    return _hash_ndarray(x_chunk).reshape((1,) * x_chunk.ndim)


def _hash_aggregate(
    x_chunk: np.ndarray,
    axis: tuple[int] | None,
    keepdims: bool,
) -> np.int64:
    """Compute a hash from a numpy array.

    This function can be applied as the final step in :func:`~dask.array.reduction`.

    Parameters
    ----------
    x_chunk :
        The array to hash.
    axis :
        Unused but required by :func:`~dask.array.reduction`.
    keepdims :
        Unused but required by :func:`~dask.array.reduction`.

    Returns
    -------
    np.int64 :
        The hash value.

    """
    (result,) = _hash_ndarray(x_chunk)
    return result


def _hash_array(a: da.Array | np.ndarray) -> np.int64:
    """Calculate a hash representation of the provided array.

    Calculates a 64-bit non-cryptographic hash of the provided array, using
    the fast ``xxhash`` hashing algorithm.

    Note that the computed hash depends on how the array is chunked.

    Parameters
    ----------
    a :
        The array that requires to have its hexdigest calculated.

    Returns
    -------
    np.int64
        The array's hash.

    """
    if isinstance(a, da.Array):
        # Use :func:`~dask.array.reduction` to compute a hash from a Dask array.
        #
        # A hash of each input chunk will be computed by the `chunk` function
        # and those hashes will be combined into one or more intermediate chunks.
        # If there are multiple intermediate chunks, a hash for each intermediate
        # chunk will be computed by the `combine` function and the
        # results will be combined into a new layer of intermediate chunks. This
        # will be repeated until only a single intermediate chunk remains.
        # Finally, a single hash value will be computed from the last
        # intermediate chunk by the `aggregate` function.
        result = da.reduction(
            a,
            chunk=_hash_chunk,
            combine=_hash_chunk,
            aggregate=_hash_aggregate,
            keepdims=False,
            meta=np.empty(tuple(), dtype=np.int64),
            dtype=np.int64,
        )
    else:
        result = _hash_aggregate(a, None, False)
    return result
