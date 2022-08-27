import argparse
import datetime
import sys
import os, django
from train import *
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,dir_path + "/..")

import time
from views import models

# parse parameters
arg = argparse.ArgumentParser()
arg.add_argument('--args', type=str, default=None)
args_dict = eval(arg.parse_args().args)

# create folders and files
output_path = dir_path + "/../static/models/" + args_dict['time'].strftime("%Y%m%d%H%M%S") + "/"
os.mkdir(output_path)
os.mkdir(output_path + "images/")
f = open(output_path + "output.log", "w")
f.close()

# train
train(epochs=int(args_dict['epochs']), batch_size=int(args_dict['batch_size']), learning_rate=float(args_dict['learning_rate']), add_conv_layer=(args_dict['lenet_structure'] != 'LeNet'), optim=args_dict['optimizer'], show_loss=True, show_accuracy=True, output_path=dir_path + "/../static/models/" + args_dict['time'].strftime("%Y%m%d%H%M%S") + "/")
os.rename(output_path + "lenet.pth", output_path + args_dict['time'].strftime("%Y%m%d%H%M%S") + "_lenet.pth")

# update database
updated_model = models.Record.objects.get(created_at=args_dict['time'])
updated_model.updated_at = datetime.datetime.now().replace(microsecond=0)
updated_model.save()