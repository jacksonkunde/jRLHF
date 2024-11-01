from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jRLHF",
    version="0.1.0",
    author="Jackson Kunde",
    author_email="jkunde@wisc.edu",
    description="A reward model trainer and rewarder module built using jtransformer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["jRLHF", "jRLHF.*"]),
    install_requires=[
        "torch==2.5.0",
        "numpy==2.1.2",
        "transformers==4.45.2",
        "datasets==3.0.2",
        "wandb==0.18.5",
        "tabulate==0.9.0",
    ],
    dependency_links=[
        "git+https://github.com/jacksonkunde/jtransformer.git#egg=jtransformer"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
