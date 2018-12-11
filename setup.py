from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pib",
    version="1.3",
    author="thanhnguyentang",
    author_email="nguyent2792@gmail.com",
    description="A Keras implementation of Parametric Information Bottleneck.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thanhnguyentang/pib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
