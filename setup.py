from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(path: str) -> List[str]:
    """
    Read the requirements file and return the list of requirements
    """
    requirements = []
    with open(path) as f:
        f.read()
        requirements = f.readlines()
        requirements = [requirement.replace("\n", "") for requirement in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="ml_e2e_package",
    version="0.0.1",
    author="Shivam",
    author_email="shivamkichitthi@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
