import re

from setuptools import find_packages, setup

project_name = 'oasis'


# auto-updating version code taken from RadVel
def get_property(prop, project):
    with open(project + "/__init__.py") as f:
        result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), f.read())
    return result.group(1)


def get_requires():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name=project_name,
    author="Edgar Salazar",
    author_email="edgarmsc@arizona.edu",
    maintainer="Bayli Hayes",
    maintainer_email="baylihayes@arizona.edu",
    version=get_property("__version__", project_name),
    description="Orbiting mass assignment scheme",
    license="GPL-3.0 license",
    url="https://github.com/baylihayes/oasis",
    packages=find_packages(exclude=['test', 'test.*']),
    python_requires='>=3.10',
    install_requires=get_requires(),
    extras_require={'test': ['pytest']},
)