import os
import sys
from setuptools import setup, find_packages


package_basename = 'PolyBin3D'
package_dir = os.path.join(os.path.dirname(__file__), package_basename)
sys.path.insert(0, package_dir)

# install package
setup(
    name=package_basename,
    packages=find_packages(),
    include_package_data=True,
)