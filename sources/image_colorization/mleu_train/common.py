"""
IMPORT COMMON LIBRARIES
"""
import os, sys, glob, tqdm, numpy as np, argparse
import argparse
import json5, shutil, subprocess, datetime, jsonpickle, random
from math import ceil

import IPython
from IPython import display, get_ipython
import ipywidgets as widgets
import base64
import pprint
import gc
import runpy, shlex

# Image Processing
import cv2

# Database
import pandas as pd

import matplotlib.pyplot as plt


# K-Fold
# + StratifiedKFold: nhu k-fold nhung bao dam phan bo cua tung class trong fold
# + K-Fold: chia deu tap du lieu thanh k phan bang nhau
from sklearn.model_selection import KFold, StratifiedKFold

# Our libraries
from prlab.tf2.utils import view_keras_model, save_keras_model, model_summary_string
from prlab.utils.visualize import visualize_logs
from prlab.tf2.envs import init_tf_environment
from prlab.utils.logs import build_log_print, print_string, print_logs_metrics
from prlab.tf2.callbacks.clr_schedule import CyclicLR
from prlab.tf2.callbacks.lr_finder import LRFinder
from prlab.tf2.callbacks.lr_schedule import step_decay_schedule, cosine_anneal_schedule


# Keras
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from tensorflow.keras.layers import Flatten, Dropout, Concatenate, BatchNormalization, Input, Convolution2D, MaxPooling2D, concatenate, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LambdaCallback
from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, SGD, Adadelta, Adam
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

# qubvel
import segmentation_models as sm

# new
try:
    from tensorflow_addons.optimizers import AdamW
except:
    pass
# try