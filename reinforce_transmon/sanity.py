# -*- coding: utf-8 -*-

# This code is part of reinforce_transmon.
#
# (C) Copyright Lucas Camponogara Viera, 2026.
#
# This code is licensed under the Apache 2.0 License.
# You may obtain a copy of the License in the root directory of this source tree.

"""reinforce_transmon 2026"""

try:
    import numpy  # noqa: F401
except ImportError:
    print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on NumPy.\n \
      ###################################\n")

try:
    import tensorflow  # noqa: F401
except ImportError:
    print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on TensorFlow 2.\n \
      ###################################\n")

try:
    import scqubits  # noqa: F401
except ImportError:
    print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on scQubits SDK\n \
      ###################################\n")
    raise
