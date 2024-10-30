from setuptools import setup, find_packages


def get_requirements(filename="requirements.txt"):
    with open(filename) as f:
        return f.read().splitlines()


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
    install_requires=get_requirements(),
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
