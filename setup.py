# setup.py

from setuptools import setup

setup(
    name='plakakia',
    version='0.2.0',
    author='Yannis Kalfas',
    author_email='kalfasyan@gmail.com',
    description='Python image tiling library for image processing, object detection, etc.',
    packages=['plakakia'],
    install_requires=[
        'numpy>=1.24.2',
        'pandas>=1.5.3',
        'matplotlib>=3.7.1',
        'opencv-python>=4.7.0.72',
        'scipy>=1.10.1',
        'seaborn>=0.12.2',
        'lxml>=4.9.2',
        'imagesize>=1.4.1',
        'psutil>=5.9.5',
        'tqdm>=4.65.0',
        'PyYAML>=6.0',
        'pyarrow>=12.0.0',
        'fastparquet>=2023.4.0',
    ],
    entry_points={
        'console_scripts': [
            'make_tiles = plakakia.make_tiles:main',
        ],
    },
)