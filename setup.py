from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    HYPHEN_E_DOT = "-e ."
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] #in this case we are replacing with \n -> "" by using list comprehension

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements



setup(
    name='Machine-Learning',
    author='Vishnu',
    author_email='vishugupta7699181@gmail.com',
    packages=find_packages(),
    version='0.0.1',
    install_requires=get_requirements("requirements.txt")
)