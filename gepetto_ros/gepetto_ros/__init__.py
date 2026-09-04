"""The planner-to-robot bridge, for an LBR arm with either hand.

IMPORTING THIS PACKAGE MAKES ``gepetto_solvers`` IMPORTABLE, and that is not
decoration -- every node here is started by ROS's system python while the solver
stack lives in a conda env, and the two are stitched together below. Read on
before removing it.

WHY PYTHONPATH IS NOT ENOUGH. The nodes run under ``/usr/bin/python3``, because
that is what ``rclpy``, ``tf2_ros`` and MoveIt were built against; ``gepetto_solvers``,
``viser`` and the compiled solver binding live in a conda env built for the same
CPython 3.10 ABI. Putting that env's ``site-packages`` on ``PYTHONPATH`` -- which
is what every launch file here does, per node, so conda never reaches ``move_group``
or RViz -- adds the directory to ``sys.path`` but does NOT make it a *site*
directory, and only a site directory has its ``.pth`` files executed.

That distinction is invisible for an ordinary install and fatal for an EDITABLE
one. ``gepetto_solvers`` is installed editable (scikit-build-core), which means
``site-packages/gepetto_solvers/`` holds nothing but the ``.so`` and a
``_editable_skbc_gepetto_solvers.pth`` points at the source tree for the Python
half. With the ``.pth`` unread, ``import gepetto_solvers`` finds the ``.so``
directory, ``gepetto_solvers.core`` does not exist, and the failure reads as a
missing module rather than as a path problem. Adding the source tree to
``PYTHONPATH`` by hand does not fix it either -- the source package then SHADOWS
the one holding the binding, and the import fails the other way round.

:func:`ensure_solvers_on_path` calls ``site.addsitedir``, which is exactly the
"make this a site directory" operation, ``.pth`` processing and all. It runs on
import of this package, before any node module has had a chance to
``import gepetto_solvers``, and it is a no-op when the solvers are already
importable -- so a workspace running everything inside the conda env, or against a
non-editable install, is unaffected.
"""

import os
import site
import sys

#: Overrides the conda prefix the solver stack is looked for in. The launch files
#: set it alongside the ``PYTHONPATH`` they inject, so the two cannot name
#: different envs.
ENV_VAR = "GEPETTO_CONDA_PREFIX"

#: Where to look when nothing says otherwise, in order. These are guesses that
#: make ``ros2 run`` work with no configuration in the layout this repo is
#: actually used in; a node started by the launch files never needs them.
_DEFAULT_PREFIXES = (
    os.path.expanduser("~/miniconda3/envs/gepetto_py10"),
    os.path.expanduser("~/anaconda3/envs/gepetto_py10"),
)


def _solvers_complete():
    """Whether ``gepetto_solvers`` is importable AND has its Python half.

    Both halves, deliberately: the failure this exists to prevent is the one
    where the top-level package imports (it is the ``.so``'s directory) and
    ``gepetto_solvers.core`` does not.

    A FAILED PROBE LEAVES NOTHING BEHIND, and that is the subtle half. With the
    conda env on ``PYTHONPATH`` -- which the launch files also set -- the name
    ``gepetto_solvers`` resolves to ``site-packages/gepetto_solvers/``, a
    directory holding only the ``.so`` and no ``__init__.py``, i.e. a NAMESPACE
    package. Importing it succeeds, caches that namespace in ``sys.modules``, and
    then no amount of later path fixing can replace it: the editable finder
    installed by :func:`ensure_solvers_on_path` is consulted only for names that
    are not already imported. The symptom is
    ``ImportError: cannot import name 'PerDiscTendonInput' from 'gepetto_solvers'
    (unknown location)`` -- "unknown location" being the namespace package saying
    it has no ``__init__``. So a probe that does not pan out purges what it
    imported, and the retry starts clean.
    """
    try:
        import gepetto_solvers  # noqa: F401
        import gepetto_solvers.core  # noqa: F401
    except ImportError:
        for name in [n for n in sys.modules
                     if n == "gepetto_solvers" or n.startswith("gepetto_solvers.")]:
            del sys.modules[name]
        return False
    return True


def ensure_solvers_on_path(prefix=None):
    """Make ``gepetto_solvers`` importable. Returns the prefix used, or None.

    Tries, in order: nothing at all (already importable), ``prefix``,
    ``$GEPETTO_CONDA_PREFIX``, then the default guesses. Never raises -- a node
    that genuinely cannot find the solvers should fail on its own ``import`` with
    the module name in the message, not here with a path in it.
    """
    if _solvers_complete():
        return None
    candidates = [prefix, os.environ.get(ENV_VAR), *_DEFAULT_PREFIXES]
    for candidate in candidates:
        if not candidate:
            continue
        packages = os.path.join(candidate, "lib",
                                f"python{sys.version_info.major}."
                                f"{sys.version_info.minor}", "site-packages")
        if not os.path.isdir(packages):
            continue
        # addsitedir, not sys.path.insert: the point is the .pth processing.
        site.addsitedir(packages)
        if _solvers_complete():
            return candidate
    return None


SOLVERS_PREFIX = ensure_solvers_on_path()
