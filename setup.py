from setuptools import setup, find_packages


setup(
    name='cuicui',
    version='0.1.0',
    description='CuiCui is designed to assit with tasks related to AI/ML',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Kle-oliver/cuicui',
    author='Kleverson Santos',
    author_email='kleverson.contact@gmail.com',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha'
    ],
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'h5py>=3.11.0,<4.0.0',
        'numpy>=2.1.1,<3.0.0',
        'pillow>=10.4.0,<11.0.0'
    ],
    extras_require={
        'test': ['pytest>=8.3.3,<9.0.0']
    },
    include_package_data=True
)
