import json
from data.base import *
from data.cityscapes_loader import cityscapesLoader

def get_loader(name):
    return {"cityscapes": cityscapesLoader,}[name]

def get_data_path(name):
    return './data/cityscapes/'
