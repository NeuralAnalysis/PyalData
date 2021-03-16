from setuptools import find_packages, setup

setup(
    name='pyaldata',
    packages=find_packages(),
    version='0.1.0',
    description='',
    author='Matt Perich, Catia Fortunato, Bence Bagi',
    #license='License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    install_requires = [
        'numpy',
        'scipy>=1.5',
        'pandas>=1.2.0',
        'scikit-learn'
    ]
)

