from setuptools import setup, find_packages

setup(
    name="sc3s", 
    version="0.0.1-dev",
    description="Python Single Cell Consensus Clustering",
    url="https://github.com/pittachalk/sc3s",
    author="Fu Xiang Quah",
    author_email="fxq20@cam.ac.uk",
    classifiers=[
       "Programming Language :: Python :: 3",
       "License :: OSI Approved :: BSD License",
       "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    python_requires=">=3.6",    
    install_requires=[
       "numpy>=1.19.1", 
       "pandas>=1.1.3", 
       "scikit-learn>=0.23.2",
       "scanpy>=1.6.0" 
  ]
)
