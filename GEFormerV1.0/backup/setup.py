
from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize(["train_optuna.py","train.py","pred.py"]))

#python setup.py build_ext --inplace