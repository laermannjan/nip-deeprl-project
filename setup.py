from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='project_framework',
      description='Framework package to evaluate custom setups on OpenAI Gym Environments.',
      long_description=readme(),
      version='0.1',
      url='http://github.com/laermannjan/nip-deeprl-project',
      author='Jan Laermann',
      author_email='laermannjan@gmail.com',
      packages=['project_framework'],
      zip_safe=False,
      install_requires=[
          'numpy',
          'gym',
          'baselines'
      ]
)
