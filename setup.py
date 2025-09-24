import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    author_email="orthogonaldfa@kavigupta.org",
    description="Learn a set of orthogonal DFAs to cover the behavior of a neural model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kavigupta/orthogonal-dfa",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch==2.7.1",
        "dconstruct==1.0.0",
        "numpy==1.23.3",
        "pythomata==0.3.2",
    ],
    # documentation
    project_urls={
        "Documentation": "https://orthogonal-dfa.readthedocs.io/en/latest/#",
    },
)
