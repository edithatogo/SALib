try:
    from numba import njit as _njit, jit as _jit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba not installed
    NUMBA_AVAILABLE = False

    def _njit(*dargs, **dkwargs):  # noqa: D401
        """Dummy decorator when numba is unavailable."""
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        def wrapper(func):
            return func

        return wrapper

    def _jit(*dargs, **dkwargs):
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        def wrapper(func):
            return func

        return wrapper


def optional_njit(*dargs, **dkwargs):
    """Return ``numba.njit`` decorator if available, else no-op."""

    def decorator(func):
        if NUMBA_AVAILABLE:
            try:
                if dkwargs.get("nopython") is False:
                    dkwargs.pop("nopython")
                    return _jit(forceobj=True, **dkwargs)(func)
                return _njit(*dargs, **dkwargs)(func)
            except Exception:
                return _jit(nopython=False)(func)
        return func

    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        func = dargs[0]
        return decorator(func)
    return decorator

__all__ = ["optional_njit", "NUMBA_AVAILABLE"]
