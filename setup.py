# This code is part of reinforce_transmon.
#
# (C) Copyright Lucas Camponogara Viera, 2026.
#
# This code is licensed under the Apache 2.0 License.
# You may obtain a copy of the License in the root directory of this source tree.

"""reinforce_transmon 2026"""

from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent.absolute()

long_description = (here / "README.md").read_text(encoding="utf-8")

VERSION = (here / "reinforce_transmon" / "VERSION.txt").read_text().strip()

REPO_URL = "https://github.com/camponogaraviera/reinforce-transmon"

setup(
    name="reinforce_transmon",
    packages=find_packages(),
    include_package_data=True,
    package_data={"reinforce_transmon": ["VERSION.txt", "src/config.cfg"]},
    version=VERSION,
    description="reinforce_transmon | Finding the Transmon Regime with DDPG Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lucas Camponogara Viera",
    author_email="vieracamponogara@gmail.com",
    license="Apache-2.0",
    platforms=[
        "Windows",
        "Linux",
    ],
    python_requires=">=3.11,<3.12",
    classifiers=[
        "Development Status :: 4 - Beta",  # https://pypi.org/classifiers/
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    keywords="RL Transmon",
    project_urls={
        "Bug Tracker": f"{REPO_URL}/issues",
        "Documentation": REPO_URL,
        "Source Code": REPO_URL,
    },
    entry_points={
        "console_scripts": [
            "reinforce_transmon=reinforce_transmon.cli:main",
        ],
    },
)
