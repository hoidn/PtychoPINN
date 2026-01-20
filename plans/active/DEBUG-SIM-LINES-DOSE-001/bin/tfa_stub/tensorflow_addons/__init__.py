"""Minimal tensorflow_addons shim for legacy dose_experiments scripts.

This package exposes only the pieces currently used by the legacy runner:

* `tensorflow_addons.image.translate`
* `tensorflow_addons.image.gaussian_filter2d`

It is intentionally tiny so we can run the old scripts without installing
the full tensorflow-addons dependency (which now hard-requires the Keras 3
package and breaks under the frozen environment policy).

The implementation prioritizes correctness over raw speed; the helpers rely
solely on TensorFlow ops so they run wherever TensorFlow itself runs.
"""

from . import image  # noqa: F401

__all__ = ["image"]
