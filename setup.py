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
    install_requires=[],
    # documentation
    project_urls={
        "Documentation": "https://orthogonal-dfa.readthedocs.io/en/latest/#",
    },
)
