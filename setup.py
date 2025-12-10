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
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch==2.7.1",
        "dconstruct==1.0.0",
        "numpy==2.3.3",
        "pythomata==0.3.2",
        "automata-lib==8.3.0",
        "frame-alignment-checks>=0.0.73",
        "coloraide",
        "h5py==3.15.1",
        "scikit-learn==1.5.1",
        "permacache==5.0.0",
    ],
    # documentation
    project_urls={
        "Documentation": "https://orthogonal-dfa.readthedocs.io/en/latest/#",
    },
)
