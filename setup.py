import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    author_email="AUTHOR_EMAIL_OF_PACKAGE",
    description="DESCRIPTION_OF_PACKAGE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GITHUB_NAME_OF_AUTHOR_OF_PACKAGE/NAME_OF_PACKAGE",
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
        "Documentation": "https://NAME_OF_PACKAGE.readthedocs.io/en/latest/#",
    },
)
