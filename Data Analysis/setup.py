import setuptools
from setuptools.command.develop import develop

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        try:
            import robusta as rst
        except: # Most likely an R package installation error
            print('Installation of robusta is not finished. Retrying...')
            import robusta as rst

setuptools.setup(
    name="mr_utils",
    version="0.0.1",
    author="Eitan Hemed",
    author_email="eitan.hemed@gmail.com",
    description="Utilities for analysis of the project",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/EitanHemed/mr-colour",
    project_urls={
        "Bug Tracker": "https://github.com/EitanHemed/mr-colour/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3",
        "Operating System :: OS Independent",
    ],
    package_dir={"mr_utils": "mr_utils"},
    packages=setuptools.find_packages(where="mr_utils"),
    python_requires="==3.9.12",
    install_requires=['robusta-stats==0.0.4', 'matplotlib==3.7.1', 'scipy==1.10.1',
                      'seaborn==0.12.2', 'tabulate==0.9.0'],
    cmdclass={
        'develop': PostDevelopCommand,
    },
)
