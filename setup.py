from setuptools import setup
  
setup(
    name='otflow-jax',
    version='0.1',
    description='Jax implementation of OT-Flow',
    author='Brian J. McDermott',
    author_email='bjmcder@gmail.com',
    packages=['otflow'],
    install_requires=[
        'jax',
        'pandas',
    ],
)