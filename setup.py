from setuptools import setup, find_packages

setup(
    name="cardionexus-ai",
    version="5.0.0",
    author="Mohamed Salih R.S.",
    author_email="salih500@gmail.com",
    description="Open-source cardiac AI platform with novel risk pathways",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Fahmu500/CardioNexus-AI",
    license="Apache-2.0",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="cardiac ai heart disease risk prediction deep-learning medical-ai",
)
