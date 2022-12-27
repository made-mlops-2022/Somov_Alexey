from setuptools import find_packages, setup

REQUIREMENTS_PATH = 'requirements.txt'


def get_requirements(path):
    with open(path, 'r') as fd:
        requirements = fd.read().splitlines()
    return requirements


setup(
    name="online_inference",
    packages=find_packages(),
    version="0.0.1",
    install_requires=get_requirements(REQUIREMENTS_PATH),
    license="MIT",
)