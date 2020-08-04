"""
stop_criteria_fn
custom_on_epoch_begin
custom_on_epoch_end
"""
import datetime, json5, os, pprint
from IPython import display
import pandas as pd
import gc, cv2

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, TensorBoard

from prlab.utils.logs import print_logs_metrics
from prlab.utils.visualize import visualize_logs

from prlab.tf2.callbacks.clr_schedule import CyclicLR
from prlab.tf2.callbacks.lr_finder import LRFinder
from prlab.tf2.callbacks.lr_schedule import step_decay_schedule, cosine_anneal_schedule
from image_colorization.datasets import view_images

def stop_criteria_fn(stop_criteria_file, params, logs, verbose = 1, print = print, **kwargs):
    """
    Checking for stop training
    stop_criteria_file: 
    {
        dice_coef: 0.93,
        bias_dice_coef: 999999,
        loss: -999999,
        bias_loss: 999999,
    }

    """
    with open(stop_criteria_file, "rt") as fp: 
        stop_criteria_cfg = json5.load(fp)
        params["stop_train_criteria"]["criteria"] = stop_criteria_cfg
    # with

    is_stop = False
    t_msg = ""

    print("+ Checking Stop Criteria: ")
    for s_criteria in stop_criteria_cfg:
        s_eval = eval("f'%s'"%(s_criteria), logs)
        r_eval = eval(s_eval)
        ret    = "--> Stopping" if r_eval==True else " --> None"
        msg = f'{s_criteria} => {s_eval} => {r_eval} {ret}'
        print(msg)
        t_msg = t_msg + msg + "\n"
        if r_eval == True: is_stop = True
    # for
    print()

    return is_stop, t_msg
    pass
# stop_criteria_fn

def custom_on_epoch_begin(self, epoch, logs, train_session, params, app_cfg, print = print, **kwargs):
    print(f"---- Start Epoch {epoch} ----")
    cur_time   = datetime.datetime.now()
    cur_period = (datetime.datetime.now() - train_session["info"]["starting_time"]).total_seconds()/60.0
    
    cur_time_s   = "{date:%y%m%d%H%M%S}".format(date = datetime.datetime.now())
    cur_period_s = f"{cur_period:.2f}"
    print(f"Current time: {cur_time_s}")
    print(f"Current period: {cur_period_s} min")
    pass
# custom_on_epoch_end

def custom_on_epoch_end(self, epoch, logs, 
            train_session, params, app_cfg, 
            print = print, **kwargs):  
    logs = logs.copy()
    print(logs)

    print(f"---- End Epoch {epoch} ----")
    cur_time   = datetime.datetime.now()
    cur_period = (datetime.datetime.now() - train_session["info"]["starting_time"]).total_seconds()/60.0
    
    cur_time_s   = "{date:%y%m%d%H%M%S}".format(date = datetime.datetime.now())
    cur_period_s = f"{cur_period:.2f}"
    print(f"Current time: {cur_time_s}")
    print(f"Current period: {cur_period_s} min")

    lr = K.eval(self.model.optimizer.lr)
    new_logs = {"epoch": epoch, "date": cur_time_s, 
                "period": cur_period_s, "lr":float(lr)}
    logs.update(new_logs)
    if len(train_session["review_columns"])==0:
        for key in logs.keys():
            train_session["review_logs"][key] = []
            train_session["review_columns"].append(key)
        # for
    # if
    for key in logs.keys(): 
        if key in train_session["review_columns"]:
            train_session["review_logs"][key].append(logs[key])
        # if
    # for

    # View log metrics
    print_logs_metrics(logs, train_session["review_logs"], print = print)
    
    # Save to log files
    df_logs = pd.DataFrame(train_session["review_logs"], columns = train_session["review_columns"])
    df_logs.to_csv(train_session["csv_logs_path"], index = False)
              
    # Visualize and save to images
    visualize_data = train_session["review_logs"]
    groups         = app_cfg["metric_visualize"]["groups"]
    figsize        = tuple(app_cfg["metric_visualize"]["figsize"])
    for name in groups:
        group = groups[name]
        save_path = f'{train_session["logs_dir"]}/logs_{name}.jpg' 
        visualize_logs(visualize_data, group, figsize, False, save_path = save_path)
        print(f'Save log visualize: {save_path}')
    # for
    
    if epoch>0 and epoch%10 == 0:
        display.clear_output()
    # if
    
    # Preview images
    train_session["data"]["preview_test_process"].save_predict(self.model, train_session, epoch)    
    # Preview valid images
    train_session["data"]["preview_valid_process"].save_predict(self.model, train_session, epoch)
    
    
    # Stopping Checking
    print("Checking stopping...")
    is_stop, msg = stop_criteria_fn(train_session["stop_criteria_file"], params, logs, print = print)
    # is_stop, msg = False, ""
    if is_stop == True:
        print(f'>> Stop by criteria')
        self.model.stop_training = True
    # if
    ## delete to stop
    if os.path.exists(train_session["stop_delete_file"]) == False:
        print(">> Stop by user!")
        self.model.stop_training = True
    # if
    pass

    print("Free Memory...")
    del df_logs
    gc.collect()
    
    print("---- END EPOCH ----")
# custom_on_epoch_end

def get_callbacks(train_session, params, app_cfg, print = print, **kwargs):
    def _model_checkpoint():
        # + ModelCheckPoint
        
        
        modekcheckpoint_template_name = params["modekcheckpoint_template_name"]
        modekcheckpoint_default_callback_info = dict(
            filepath = f'{train_session["checkpoints_dir"]}/{modekcheckpoint_template_name}',
            monitor  = "val_loss", 
            verbose  = 1, 
            save_best_only=True, 
            save_weights_only=False, # only weight, 
            mode="min", # min, max, auto
            # period=1  # bao nhieu epoch moi xem xet 1 lan
            save_freq="epoch",  # bao nhieu epoch moi xem xet 1 lan
        )
        
        cbl = []
        for key in params["modekcheckpoint_cfg_list"].keys():
            modekcheckpoint_callback_info = {}
            modekcheckpoint_callback_info.update(**modekcheckpoint_default_callback_info)
            modekcheckpoint_callback_info.update(**params["modekcheckpoint_cfg_list"][key])
            modekcheckpoint = ModelCheckpoint(**modekcheckpoint_callback_info)
            cbl.append(modekcheckpoint)
            print(f'+ {key}:\n{pprint.pformat(modekcheckpoint_callback_info)}')
        # for
        return cbl
    # _model_checkpoint
    
    print("Callbacks: ")

    # callbacks
    callbacks = []
    
    # + ModelCheckpoint
    print("+ ModelCheckpoint Callbacks")
    callbacks.extend(_model_checkpoint())

    # + Early Stopping: Stop training when a monitored quantity has stopped improving.
    if params["early_stopping"]>0:
        earlystopping_callback = EarlyStopping(monitor='val_loss', patience= params["early_stopping"], mode="min", verbose=1)
        callbacks.append(earlystopping_callback)
        
        print(f'+ Early Stopping Callbacks: {params["early_stopping"]}, va_loss, min')
    # if

    # + CSV Logs
    print(f'+ csv logs path: {train_session["csv_logs_path"]}')

    # + Callback List Info
    cbl_info = []
    cbl_info.extend(app_cfg['callbacks_list'][params['type_callbacks_common']])
    cbl_info.extend(app_cfg['callbacks_list'][params['type_callbacks_others']])
    print("\nCallback List Info: ")
    for cb in cbl_info:
        cb_fn = eval(cb["fn"])
        print(f'+ Callback Name: {cb["name"]}')
        print(f'+ Callback Fn  :\n{cb["fn"]}')
        print(f'--> eval:\n{cb_fn}')
        callbacks.append(cb_fn)
        print()
    # for

    # + Custom Callback
    custom_callbacks = LambdaCallback()
    custom_callbacks.on_epoch_begin = lambda epoch, logs: \
        custom_on_epoch_begin(custom_callbacks, epoch, logs, \
            train_session = train_session, params = params, app_cfg = app_cfg, 
            print = print)
    custom_callbacks.on_epoch_end = lambda epoch, logs: \
        custom_on_epoch_end(custom_callbacks, epoch, logs, \
            train_session = train_session, params = params, app_cfg = app_cfg, 
            print = print)
    callbacks.append(custom_callbacks)
    print()

    return callbacks
# get_callbacks
