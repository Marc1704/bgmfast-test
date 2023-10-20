from setuptools import setup, find_packages

setup(
    name="bgmfast",
    version="0.0.1",
    autor="Marc del Alcázar i Julià",
    author_email="mdelalju@fqa.ub.edu",
    url="",
    readme="README.md",
    description="The Besançon Galaxy Model Fast Approximate Simulations",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas", "numpy", "astropy", "pyspark"],
)
