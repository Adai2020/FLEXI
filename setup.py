from setuptools import setup, find_packages

setup(
    name="FLEXI",
    packages=find_packages(), 
    python_requires=">=3.9",
    install_requires=[
        "numpy==1.26.0",
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
        "pandas==2.2.2",
        "seaborn==0.13.2",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "matplotlib==3.8.2",
        "librosa==0.10.1",
    ],
)