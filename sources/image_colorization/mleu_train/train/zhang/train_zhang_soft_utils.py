"""
stop_criteria_fn
custom_on_epoch_begin
custom_on_epoch_end
"""
import datetime, json5, os, builtins, cv2
from IPython import display
import pandas as pd

import gc, shutil, subprocess, jsonpickle
from image_colorization.datasets import view_images

class PreviewImageProcess(object):
    def __init__(self, preview_gen, 
                       preview_name = "preview_valid", # preview_valid, preview_test
                       decode_version = 0, # version 1, usegpu=True or version 0
                       decode_usegpu=False, 
                       print = print):
        self.preview_gen = preview_gen
        self.preview_name = preview_name
        self.preview_gen.reset()
        self.x, self.y = next(self.preview_gen)
        self.i_rgb = self.preview_gen.decode_batch_image(self.x, self.y)
        
        self.decode_version = decode_version
        self.decode_usegpu = decode_usegpu

        print(f"PreviewImageProcess for {self.preview_name}: ")
        print(f"+ Decode verions: {self.decode_version}")
        print(f"+ Decode usegpu: {self.decode_usegpu}")
        print()
    # __init__
    
    def __call__(self, model, batch_size = 4):
        self.y_pred = model.predict(self.x, batch_size = batch_size)
        self.pred_i_rgb = self.preview_gen.decode_batch_image(self.x, self.y_pred, version = self.decode_version, usegpu=self.decode_usegpu)
        return self.pred_i_rgb
    # __call__

    def view_info(self, print = print):
        print(f"Init {self.preview_name} images: ")
        if self.i_rgb is not None:
            print(f'+ Soft Length: {self.i_rgb.shape}')
        print()
    # view_info

    def save_predict(self, model, train_session, epoch, print = print, is_show = True):
        true_preview_rgb = self.i_rgb
        pred_preview_rgb = self(model)

        result = []

        print(f"Predict {self.preview_name} images: ")   
        
        psnr_soft        = cv2.PSNR(true_preview_rgb, pred_preview_rgb)
        print(f'PSNR {self.preview_name} Soft Preview: {psnr_soft}')
        print()
    
        view_images(pred_preview_rgb, view_ids = range(len(pred_preview_rgb)), cols = 4, figsize=(8, 8), 
                    save_path=f'{train_session["logs_dir"]}/previews/{self.preview_name}_soft_images_{epoch:04d}_{psnr_soft:.2f}.jpg', is_show = False)
        result.append(psnr_soft)                        
        return result
        pass
    # save_predict

    def save_ground_truth(self, train_session, print = print, is_show = True):
        print(f'Saving {train_session["logs_dir"]}/{self.preview_name}_images.jpg')
        view_images(self.i_rgb, view_ids = range(len(self.i_rgb)), cols = 4, figsize=(8, 8), 
                    save_path=f'{train_session["logs_dir"]}/{self.preview_name}_images.jpg', is_show = is_show)
        print()            
        pass
    # save_ground_truth    
# PreviewImageProcess

def start_train_info(train_session, print = print, **kwargs):
    # Train model
    print("Training")
    starting_time = datetime.datetime.now()
    if train_session.get("info") is None: train_session["info"] = {}
    train_session["info"]["starting_time"] = starting_time
    train_session["info"]["s_starting_time"] = f'{train_session["info"]["starting_time"]: %Y-%m-%d %H:%M:%S}'
    print(f'+ starting Time: {train_session["info"]["starting_time"]}')
    print()
# start_train_info

def copy_train_files(copy_files, train_session, params, **kwargs):
    print("Backup files: ")
    for from_file in copy_files:
        filename = os.path.basename(from_file)
        print(f"+ Processing [{filename}]")
        if os.path.exists(from_file) == True:
            print(f'  * Copy {os.path.relpath(from_file, start=params["root_dir"])} --> {os.path.relpath(train_session["runtime_dir"], start=params["root_dir"])}/{filename}')
            shutil.copyfile(from_file, f'{train_session["runtime_dir"]}/{filename}')
        #
        if filename.endswith("ipynb") == True and os.path.exists(from_file) == True:
            print(f'  * Convert {train_session["runtime_dir"]}/{filename}: ', end="")
            filedest = f'{train_session["runtime_dir"]}/{filename}'
            query = f"jupyter nbconvert \"{filedest}\""
            response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read().decode("utf-8")
            if response=="": 
                print("Success!")
            else:
                print("Failed!")
        # if
    # for
    print()
# copy_train_files

def end_train_info(train_session, print = print, **kwargs):
    train_time_span = (datetime.datetime.now() - train_session["info"]["starting_time"])
    train_session["info"]["stopping_time"]         =  datetime.datetime.now()
    train_session["info"]["stopping_time_s"]       =  f'{train_session["info"]["stopping_time"]: %Y-%m-%d %H:%M:%S}' 
    train_session["info"]["train_total_seconds"]   = train_time_span.total_seconds()/60.0
    train_session["info"]["train_total_seconds_s"] = f'{train_session["info"]["train_total_seconds"]: .2f} min'

    print("+ stopping_time: ", train_session["info"]["stopping_time"])
    print("+ train_total_seconds: ", train_session["info"]["train_total_seconds"])

    print("Finish train!")
    with open(f'{train_session["runtime_dir"]}/finish.txt', 'wt') as f:
        f.writelines("Finish trained!")
    # with
    pass
# def

def dump_train_info(params, app_cfg, train_session, print = print, verbose = 0, **kwargs):
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle.ext.pandas as jsonpickle_pandas

    jsonpickle_numpy.register_handlers()
    jsonpickle_pandas.register_handlers()
    jsonpickle.set_encoder_options('json', sort_keys=False, indent = 4)
    jsonpickle.set_preferred_backend('json')

    print("Dump train information.")

    print("+ Params:")
    if params.get("data") is not None: params.pop("data"); # big and too difficult for dumps
    with open(f'{train_session["runtime_dir"]}/params.json', "wt") as f:
        frozen = jsonpickle.encode(params)
        if verbose == 1: print(frozen)
        f.writelines(frozen)
    # with
    print()

    print("+ Train_session:")
    if train_session.get("data") is not None: train_session.pop("data"); # big and too difficult for dumps
    with open(f'{train_session["runtime_dir"]}/train_session.json', "wt") as f:
        frozen = jsonpickle.encode(train_session)
        if verbose == 1: print(frozen)
        f.writelines(frozen)
    # with
    print()

    print("+ App_cfg:")
    if app_cfg.get("params") is not None: app_cfg.pop("params"); # not neccessary
    if app_cfg.get("data") is not None: app_cfg.pop("data"); # not neccessary
    with open(f'{train_session["runtime_dir"]}/app_cfg.json', "wt") as f:
        frozen = jsonpickle.encode(app_cfg)
        if verbose == 1: print(frozen)
        f.writelines(frozen)
    # with
    print()
# dump_train_info