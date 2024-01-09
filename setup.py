from setuptools import setup, find_packages

setup(
    name='pointfusion',
    version='0.1.0',
    author='Mark Robinson',
    author_email='mdrobinson@wpi.edu',
    description='A package for point cloud data fusion',
    packages=find_packages(),
    install_requires=[ ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)