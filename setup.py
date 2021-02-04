from setuptools import setup, find_packages

setup(
    name="sc3s", 
    version="0.3.0",
    description="Python Single Cell Consensus Clustering",
    author="Fu Xiang Quah",
    packages=find_packages(),
    install_requires=[
       "numpy>=1.19.1", 
       "pandas>=1.1.3", 
       "scikit-learn>=0.23.2"
    ],
    classifiers=[
       "Programming Language :: Python :: 3",
       "License :: OSI Approved :: BSD License",
       "Operating System :: OS Independent"
    ],
    python_requires=">=3.6"    
)
