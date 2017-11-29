import os
import sys

print("Found Python version: {}".format(sys.version))

tools_path = 'D:\\home\\site\\tools'
if not sys.version.startswith('3.6'):

    # in python 2.7
    import urllib
    print('Installing Python Version 3.6.3')

    from zipfile import ZipFile

    if not os.path.exists(tools_path):
        os.makedirs(tools_path)
        print("Created [{}]".format(tools_path))

    python_url = 'https://apmlstor.blob.core.windows.net/wheels/python361x64.zip'
    python_file = os.path.join(tools_path, 'python.zip')
    urllib.urlretrieve(python_url, python_file)
    print("Downloaded Python 3.6.3")

    python_zip = ZipFile(python_file, 'r')
    python_zip.extractall(tools_path)
    python_zip.close()
    print("Extracted Python to [{}]".format(tools_path))

    print("Please rerun this function again to install required pip packages")
    sys.exit(0)

# in python 3.6
print("Installing packages")

import pip

def install_package(package_name):
    pip.main(['install', package_name])

install_package('https://apmlstor.blob.core.windows.net/wheels/numpy-1.13.1%2Bmkl-cp36-cp36m-win_amd64.whl')
install_package('https://apmlstor.blob.core.windows.net/wheels/pandas-0.20.3-cp36-cp36m-win_amd64.whl')
install_package('https://apmlstor.blob.core.windows.net/wheels/scipy-0.19.1-cp36-cp36m-win_amd64.whl')
install_package('https://apmlstor.blob.core.windows.net/wheels/scikit_learn-0.18.2-cp36-cp36m-win_amd64.whl')


install_package('https://apmlstor.blob.core.windows.net/wheels/gensim-2.3.0-cp36-cp36m-win_amd64.whl')
install_package('https://apmlstor.blob.core.windows.net/wheels/nltk-3.2.4-py2.py3-none-any.whl')
install_package('langid')
install_package('pymorphy2')

print("All packages were installed")

print("Configuring NLTK")

import nltk
nltk_path = os.path.abspath(os.path.join('..', 'lib', 'nltk_data'))
if not os.path.exists(nltk_path):
    os.makedirs(nltk_path)
    print("INFO: Created {0}".format(nltk_path))
    
nltk.download('punkt', download_dir=os.path.join('..', 'lib', 'nltk_data'))
nltk.download('stopwords', download_dir=os.path.join('..', 'lib', 'nltk_data'))

print("Setup is finished")