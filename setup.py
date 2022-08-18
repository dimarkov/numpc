# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['numpc']

package_data = \
{'': ['*']}

install_requires = \
[]

setup_kwargs = {
    'name': 'numpc',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'dimarkov',
    'author_email': '5038100+dimarkov@users.noreply.github.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<3.11',
}


setup(**setup_kwargs)

