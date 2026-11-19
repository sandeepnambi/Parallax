from setuptools import setup, find_packages

setup(
    name="parallax",
    version="0.2.0",
    description="Claim-Level Superseded-Evidence Detection in Retrieval-Augmented Generation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["verbalizer", "dataset_builder", "benchmark_cleaner_scaler"],
    install_requires=[
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "neural": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
        ],
        "test": [
            "pytest>=7.0.0",
        ],
    },
    python_requires=">=3.9",
)
