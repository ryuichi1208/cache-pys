from setuptools import setup, find_packages

setup(
    name="cache-pys",
    version="0.1.0",
    description="Advanced caching library with multiple TTL algorithms",
    author="ryuichi1208",
    author_email="ryucrosskey@gmail.com",
    url="https://github.com/ryuichi1208/cache-pys",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
)
