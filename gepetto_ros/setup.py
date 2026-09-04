from setuptools import find_packages, setup

package_name = 'gepetto_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joe',
    maintainer_email='joe.liechty6@gmail.com',
    description='Planner-to-robot bridge for the LBR arm with either hand.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'viz_node = gepetto_ros.viz_node:main',
            'executor_node = gepetto_ros.executor_node:main',
            'home_node = gepetto_ros.home_node:main',
            # Bring-up tools. Ordinary console scripts rather than a test/
            # directory: they poke a LIVE system and are run the way every other
            # node is.
            'play_client = gepetto_ros.play_client:main',
            'fake_cell = gepetto_ros.fake_cell:main',
        ],
    },
)
