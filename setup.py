from setuptools import setup

setup(name='mlecon',
      version='0.1.9',
      description='Machine Learning for Economic Modeling',
      url='',
      author='Victor Duarte',
      author_email='victor.duarte@sloan.mit.edu',
      license='',
      package_dir={'': '.'},
      package_data={'':
                    ['mlecon_compiled.cpython-36m-darwin.so',
                     'mlecon_compiled.cpython-36m-x86_64-linux-gnu.so',
                     'mlecon_compiled.cp36-win_amd64.pyd']},
      install_requires=[
          'tensorflow>=1.10',
          'progressbar2',
      ],
      python_requires='>=3.6.3',
      packages=['mlecon'],
      zip_safe=False)
