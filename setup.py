from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="searchrankmaster",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning-powered search ranking system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/search-rank-master",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "srm-train=searchrankmaster.cli:train",
            "srm-eval=searchrankmaster.cli:evaluate",
            "srm-serve=searchrankmaster.cli:serve",
        ],
    },
)
