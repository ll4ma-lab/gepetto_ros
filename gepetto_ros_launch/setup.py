import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'gepetto_ros_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joe',
    maintainer_email='joe.liechty6@gmail.com',
    description='Launch files for the gepetto planner-to-robot bridge.',
    license='Apache-2.0',
    entry_points={'console_scripts': []},
)
