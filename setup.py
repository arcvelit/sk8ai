from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# Setting up
setup(
    name="Sk8ai",
    version="1.0",
    author="arcvelit (Louka F.s.)",
    author_email="<louka.f.s@gmail.com>",
    description="Modular neural network model",
    long_description_content_type="text/markdown",
	long_description='Modular neural network for classification, regression and other cool stuff.',
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'ai', 'neural networks', 'numpy'],
    classifiers=[
        "Development Status :: Done ",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
