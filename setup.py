import setuptools

with open('README.md','r') as f:
    long_description = f.read()

setuptools.setup(
    name = 'Residual-CNN-for-MNIST-classification',
    version = '0.0.1',
    author = 'eric102004',
    author_email = 'h511163@nehs.hc.edu.tw',
    description='A Pytorch implementation of CNN with residual blocks but no fully connected layers.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/eric102004/Residual-CNN-for-MNIST-classification.git',
    packages=setuptools.find_packages(),
    )
