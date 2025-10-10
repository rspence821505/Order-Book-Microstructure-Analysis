### **setup.py**

from setuptools import setup, find_packages

setup(
    name="lob_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "hmmlearn>=0.2.7",
        "shap>=0.41.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "pyarrow>=8.0.0",  # for parquet
    ],
    python_requires=">=3.9",
    author="Your Name",
    author_email="rylan.spence@utexas.edu@example.com",
    description="LOB feature engineering and tree-based prediction pipeline",
    url="https://github.com/rspence821505/Order-Book-Microstructure-Analysis",
)
