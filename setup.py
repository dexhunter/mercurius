import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setuptools.setup(
    name="mercurius",
    version="0.0.1",
    author="Dixing (Dex) Xu",
    author_email="dixingxu@gmail.com",
    description="Yet Another Portfolio Management Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dexhunter/mercurius-dev",
    packages=setuptools.find_packages(),
    install_requires=reqs,
)
