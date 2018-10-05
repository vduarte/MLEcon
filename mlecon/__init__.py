import imp
import sys
path = imp.find_module("mlecon")[1]
sys.path.append(path)
from mlecon_compiled import *

