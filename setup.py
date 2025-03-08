from setuptools import setup, find_packages

setup(
    name='my_bpm',
    version='0.1.0',
    description='A BPM simulation library for integrated photonics',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/jwt625/bpm',  # adjust as needed
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ],
)
