from setuptools import setup, find_packages

setup(
    name="sc3s", 
    version="0.0.1dev9",
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
       "numpy>=1.19.2,<2",
       "pandas>=1.1.5",
       "scikit-learn>=0.23.2",
       "scanpy>=1.7.0",
       "six>=1.15.0",
       "setuptools>=52.0.0",
       "joblib>=1.0.0",
       "h5py >=3.1.0",
       "seaborn",
       "scipy",
       "python-dateutil >=2.8.1",
  ]
)
