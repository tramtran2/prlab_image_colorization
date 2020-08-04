#!/usr/bin/env python
# coding: utf-8

# # TRAIN ZHANG

from __future__ import absolute_import, division, print_function
from IPython import get_ipython

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from config_init import *
print_init_info()
from image_colorization.mleu_train.train.zhang.config import *
load_config(globals())


from image_colorization.mleu_train.common import *
from image_colorization.datasets import view_images

from image_colorization.mleu_train.datasets.dataset_general_class import ClassGeneralDataset
from image_colorization.mleu_train.dataaug import colorized_train_aug, colorized_valid_aug
from image_colorization.mleu_train.dataaug import tf_normal_preprocessing_rgb_fn, tf_normal_postprocessing_rgb_fn

from image_colorization.mleu_train.dataload_colorized_soft import SoftColorizedDataLoader
from image_colorization.mleu_train.datagen_colorized_soft import SoftColorizedDataGenerator
from image_colorization.mleu_train.losses import *
from image_colorization.mleu_train.models.build_models import FactoryModels

from image_colorization.mleu_train.train.zhang.train_zhang_soft_callbacks import *
from image_colorization.mleu_train.train.zhang.train_zhang_soft_utils import *

from prlab.contrib.keras_places365 import decode_prediction


# ## 1. Setup Environments

def parse_args():
    global parser, args
    
    # Parameters

    # parser
    parser = argparse.ArgumentParser(description='Zhang Colorization Model')

    # + app config
    parser.add_argument('--app_base', type=str, default='{current_dir}/train_zhang_soft')
    parser.add_argument('--app_cfg', type=str,  default="{current_dir}/train_zhang_soft.json")
    parser.add_argument('--app_bak', type=str,  default="[f'{current_dir}/train_zhang_soft_*.*']")

    args, _ = parser.parse_known_args()
# parse_args

def load_args():
    global args, app_cfg_path, app_cfg, params, print, print_logs, current_dir
    
    # app_cfg
    app_cfg_path = eval("f'%s'"%args.app_cfg)
    with open(app_cfg_path, 'rt') as file: app_cfg = json5.load(file)
    
    # params
    params  = app_cfg["params"]
    # + merge args with json
    params_args = vars(args)
    for key in params_args: 
        if params_args[key] is not None or params.get(key) is None: params[key] = params_args[key] 
    # for
    parse_json_config(app_cfg, globals(), locals(), remove_meta=True)
    
    # Project configuration
    if os.path.exists(params["runtime_dir"]) == False: os.makedirs(params["runtime_dir"])

    # + Logs
    params["runtime_log_file"] = f'{params["runtime_dir"]}/logs.txt'
    print_logs = build_log_print(params["runtime_log_file"], print, mode = "at")
    print = print_logs["print"]
    
    print("Config Information: ")
    print("parameters: ")
    for key in params.keys(): print("+ %s: %s"%(key,params[key]))
    print()
# load_args

parse_args(), load_args();


# Choose gpus, init session
init_tf_environment(init_gpus = params["gpus"], verbose = 1, print = print)

if params["disable_eager_execution"] == True:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
# if
if params["reset_default_graph"] == True:
    tf.compat.v1.reset_default_graph()
# if
print()


if params["verbose"] == 1: raise Exception("Exit Training", "Setup Environment")


# ## 2. Setup Data

app_cfg["db_info_list"] = load_json_config(params["db_info_list_path"], globals(), locals(), remove_meta=True)
params["db_info"] = app_cfg["db_info_list"][params["type_db_info"]]["info"]
params["db_scheme"] = eval("f'%s'"%app_cfg["db_info_list"][params["type_db_info"]]["scheme"], globals(), locals())


print("Loading dataset: ")
print(f'+ db_info:\n{pprint.pformat(params["db_info"])}')
print(f'+ db_scheme: {pprint.pformat(params["db_scheme"])}')
print()


ds_data = ClassGeneralDataset(**params["db_info"])
ds_data.load_scheme(params["db_scheme"])
print()

ds_data.view_summaries()
print()


ds_preview = ClassGeneralDataset(**app_cfg["db_info_list"][params["preview_db_info"]]["info"])
print(f'Preview dataset: \n{pprint.pformat(app_cfg["db_info_list"][params["preview_db_info"]]["info"])}')
print(f'Preview data: {params["preview_db_type"]}')
print()

ds_preview.load_scheme()
print()

ds_preview.view_summaries()
print()


# ## 3. Setup Loader

# ### Data Augmentation

params.update(app_cfg["augmentation_list"][params["type_aug"]])
params["train_transforms"] = eval(params["train_aug"])
params["valid_transforms"] = eval(params["valid_aug"])


print("Data Augmentation: ")
for key in ["type_aug", "train_aug", "valid_aug"]: print(f"+ {key}: {params[key]}")
print(f'+ train_transforms: \n{pprint.pformat(params["train_transforms"])}')
print(f'+ valid_transforms: \n{pprint.pformat(params["valid_transforms"])}')
print()


# ### Dataloader

params["dataloader_output_type"] = app_cfg["dataloader_output_list"][params["type_dataloader_output"]]["type"]
params["dataloader_output_cfg"]  = app_cfg["dataloader_output_list"][params["type_dataloader_output"]]["cfg"]


train_loader = SoftColorizedDataLoader(x_data = ds_data.x_train, 
                                       transforms=params["train_transforms"], 
                                       capacity = 0,
                                                                
                                       output_cfg  = params["dataloader_output_cfg"], 
                                       output_size = tuple(params["dataloader_output_size"]),

                                       verbose = 1)
    
valid_loader = SoftColorizedDataLoader(x_data = ds_data.x_valid, 
                                       transforms=params["train_transforms"], 
                                       capacity = 0,
                                                                
                                       output_cfg  = params["dataloader_output_cfg"], 
                                       output_size  = tuple(params["dataloader_output_size"]),

                                       verbose = 1, )

preview_valid_loader = SoftColorizedDataLoader(x_data = ds_data.x_valid, 
                                               transforms=params["train_transforms"], 
                                               capacity = params["preview_images"],
                                                                
                                               output_cfg  = params["dataloader_output_cfg"], 
                                               output_size  = tuple(params["dataloader_output_size"]),

                                               verbose = 1, )

preview_loader = SoftColorizedDataLoader(x_data = ds_preview.__dict__[params["preview_db_type"]], 
                                         transforms=params["valid_transforms"], 
                                         capacity = params["preview_images"],
                                                  
                                         output_cfg  = params["dataloader_output_cfg"], 
                                         output_size  = tuple(params["dataloader_output_size"]),

                                         verbose = 1, )


print("Train Loader Information:")
print(f'+ Length: {len(train_loader)}')
train_loader.view_image(10, only_info=True)
print()

print("Valid Loader Information:")
print(f'+ Length: {len(valid_loader)}')
valid_loader.view_image(10, only_info=True)
print()

print("Preview Valid Loader Information:")
print(f'+ Length: {len(preview_valid_loader)}')
preview_valid_loader.view_image(10, only_info=True)
print()

print("Preview Loader Information:")
print(f'+ Length: {len(preview_loader)}')
preview_loader.view_image(10, only_info=True)
print()


# ### Datagen

params["datagen_processing_image_fn"] = dict([(k, eval(v, globals(), locals())) for k, v in app_cfg["dataloader_output_list"][params["type_dataloader_output"]]["processing_image_list"].items()])

print("Data Generator: ")
print(f'+ Batch size: {params["batch_size"]}')
print(f'+ processing_image_list: \n{pprint.pformat(app_cfg["dataloader_output_list"][params["type_dataloader_output"]]["processing_image_list"])}')
print(f'--> \n{pprint.pformat(params["datagen_processing_image_fn"])}')


train_gen = SoftColorizedDataGenerator(dataloader=train_loader, 
                                       batch_size=params["batch_size"], 
                                       shuffle=True, 
                                          
                                       x_preprocessing_image_fn=params["datagen_processing_image_fn"]["x_pre_fn"],
                                       x_postprocessing_image_fn=params["datagen_processing_image_fn"]["x_post_fn"],)

valid_gen = SoftColorizedDataGenerator(dataloader=valid_loader, 
                                       batch_size=params["batch_size"], 
                                       shuffle=False, 
                                          
                                       x_preprocessing_image_fn=params["datagen_processing_image_fn"]["x_pre_fn"],
                                       x_postprocessing_image_fn=params["datagen_processing_image_fn"]["x_post_fn"],)

preview_valid_gen = SoftColorizedDataGenerator(dataloader=preview_valid_loader, 
                                               batch_size=params["preview_images"], 
                                               shuffle=False, 
                                     
                                               x_preprocessing_image_fn=params["datagen_processing_image_fn"]["x_pre_fn"],
                                               x_postprocessing_image_fn=params["datagen_processing_image_fn"]["x_post_fn"],)

preview_gen = SoftColorizedDataGenerator(dataloader=preview_loader, 
                                         batch_size=params["preview_images"], 
                                         shuffle=False, 
                                     
                                         x_preprocessing_image_fn=params["datagen_processing_image_fn"]["x_pre_fn"],
                                         x_postprocessing_image_fn=params["datagen_processing_image_fn"]["x_post_fn"],)


print("Train Generator: ")
print(f'+ Length: {len(train_gen)}')
x_batch, y_batch = next(train_gen)
train_gen.view_batch(x_batch, y_batch, only_info=True)
print()

print("Valid Generator: ")
print(f'+ Length: {len(valid_gen)}')
x_batch, y_batch = next(valid_gen)
valid_gen.view_batch(x_batch, y_batch, only_info=True)
print()

print("Preview Valid Generator: ")
print(f'+ Length: {len(preview_valid_gen)}')
x_batch, y_batch = next(preview_valid_gen)
preview_valid_gen.view_batch(x_batch, y_batch, only_info=True)
print()

print("Preview Generator: ")
print(f'+ Length: {len(preview_gen)}')
x_batch, y_batch = next(preview_gen)
preview_gen.view_batch(x_batch, y_batch, only_info=True)
print()


# ## 4. Setup Training

# Training Configuration
# train_session
train_session = dict(
    info                = {},
    data                = {},
    runtime_dir         = params["runtime_dir"],
    logs_dir            = params["runtime_dir"] + "/logs",
    checkpoints_dir     = params["runtime_dir"] + "/checkpoints",
    tensorboard_dir     = params["runtime_dir"] + "/tensorboard", 
    weight_dir          = params["runtime_dir"] + "/weights"
)
for key in ["runtime_dir", "logs_dir", "checkpoints_dir", "tensorboard_dir", "weight_dir"]:
    if not os.path.exists(train_session[key]): os.makedirs(train_session[key])
# for

print("train_session: ")
for key in train_session.keys(): print("+ %s: %s"%(key,train_session[key]))
print()


params["steps_per_epoch"]  = ceil(len(train_loader) / params["batch_size"]) if params["steps_per_epoch"] is None or params["steps_per_epoch"]==0 else params["steps_per_epoch"]
params["validation_steps"] = ceil(len(valid_loader) / params["batch_size"]) if params["validation_steps"] is None or params["validation_steps"]==0 else params["validation_steps"]
params["input_shape"]      = (params["image_size"], params["image_size"], 1)

print("Training info: ")
for key in ["batch_size", "final_epoch", "steps_per_epoch", "validation_steps", "input_shape", "best_point", "init_model"]:
    print("+ %s: %s"%(key, params[key]))
# for
print()


# ### Callbacks

# when stop training automatically or manually
train_session["stop_delete_file"]   = f'{train_session["runtime_dir"]}/{params["stop_train_criteria"]["stop_delete_file"]}'
train_session["stop_criteria_file"] = f'{train_session["runtime_dir"]}/{params["stop_train_criteria"]["stop_criteria_file"]}'

with open(train_session["stop_delete_file"], "wt") as f:
    f.writelines("Delete to stop training")
# with
with open(train_session["stop_criteria_file"], "wt") as f:
    f.writelines(json5.dumps(params["stop_train_criteria"]["criteria"], indent = 4))
    pass
# with

print("Stop training criteria: ")
print(f'+ stop_delete_file: {os.path.relpath(train_session["stop_delete_file"], start=exp_dir)}')
print(f'+ stop_criteria_file: {os.path.relpath(train_session["stop_criteria_file"], start=exp_dir)}')
print(f'{json5.dumps(params["stop_train_criteria"]["criteria"], indent = 4)}')
print()


is_show = False
train_session["data"]["preview_test_process"] = PreviewImageProcess(preview_gen, preview_name = "preview_test")
train_session["data"]["preview_test_process"].view_info()
train_session["data"]["preview_test_process"].save_ground_truth(train_session, is_show  = is_show)

train_session["data"]["preview_valid_process"] = PreviewImageProcess(preview_valid_gen, preview_name = "preview_valid")
train_session["data"]["preview_valid_process"].view_info()
train_session["data"]["preview_valid_process"].save_ground_truth(train_session, is_show  = is_show)


train_session["csv_logs_path"] = f'{train_session["logs_dir"]}/{params["csv_logs_name"]}'

train_session["review_logs"] = {}
train_session["review_columns"] = []

app_cfg["metric_visualize"] = app_cfg["metric_visualize_list"][params["type_metric_visualize"]]

params["modekcheckpoint_template_name"] = app_cfg["modekcheckpoint_list"][params["type_modekcheckpoint"]]["modekcheckpoint_template_name"]
params["modekcheckpoint_cfg_list"] = dict([(cfg, app_cfg["modekcheckpoint_list"]["cfg_list"][cfg])                                       for cfg in app_cfg["modekcheckpoint_list"][params["type_modekcheckpoint"]]["cfg_list"]
                                     ])

parse_json_item(params, ['modekcheckpoint_template_name'], "init_eval_f", globals(), locals())
callbacks = get_callbacks(train_session=train_session, params = params, app_cfg=app_cfg, print = print)


# ### Models

model_list = load_json_config(params["model_list_path"], globals(), locals(), remove_meta=True)
if app_cfg.get("model_list") is not None:     
    for key in app_cfg["model_list"]:
        if model_list.get(key) is None: model_list[key] = {}
        model_list[key].update(app_cfg["model_list"][key])
    # for
# if
app_cfg["model_list"] = model_list

train_session["model_name"] = app_cfg["model_list"]["base_model"][params["type_model"]]["name"]
train_session["model_cfg"]  = app_cfg["model_list"]["base_model"][params["type_model"]]["cfg"]
train_session["model_type"] = app_cfg["model_list"]["base_model"][params["type_model"]]["type"]
train_session["model_update_cfg"] = app_cfg["model_list"]["update_model"][params["type_model_update"]]
train_session["model_cfg"].update(**train_session["model_update_cfg"])


from image_colorization.mleu_train.models.build_models import FactoryModels
# Models
K.clear_session()

print("Load model")
print(f'+ Model type: {train_session["model_type"]}')
print(f'+ Model name: {train_session["model_name"]}')

result_model = FactoryModels.create(
    model_type = train_session["model_type"],
    model_name = train_session["model_name"], 
    model_cfg  = train_session["model_cfg"], 
    params     = params)

model = result_model["model"]
print(f'+ Model cfg : {pprint.pformat(train_session["model_cfg"])}')

if params["init_model"] is not None and params["init_model"] != "" and os.path.exists(params["init_model"])==True:
    print(f'Init model: {params["init_model"]}')
    model.load_weights(params["init_model"])
# if

print()


print("Save model: ")
save_keras_model(model, save_dir =train_session["runtime_dir"])
f = f'{train_session["runtime_dir"]}/model.txt'
get_ipython().system('cat $f')
print()


# ### Optimization/Loss/Metrics

categorical_crossentropy_color = build_categorical_crossentropy_color_loss(params["prior_factor_path"])
categorical_crossentropy_color_v1 = build_categorical_crossentropy_color_loss_v1(params["prior_factor_path"])
print(f'prior_factor_path: {params["prior_factor_path"]}')
print()


# compile model info
compile_model_info = dict(
    # loss
    loss = app_cfg["loss_list"][params["type_loss"]]["loss"],
    loss_weights = app_cfg["loss_list"][params["type_loss"]]["loss_weights"],
    # optimizer
    optimizer    = app_cfg["opt_list"][params["type_opt"]],
    # metrics
    metrics = app_cfg["loss_list"][params["type_loss"]]["metrics"],
)
print("compile_model_info: ")
print(f'+ learning rate: {params["learning_rate"]}')
for key in compile_model_info.keys(): # ["loss", "loss_weights", "metrics"]: 
    print(f'+ {key}: \n{compile_model_info[key]}')
    if compile_model_info[key] is not None and compile_model_info[key] !="":
        compile_model_info[key] = eval(compile_model_info[key])
    print(f'--> eval: {pprint.pformat(compile_model_info[key])}')
# for
print()


print("Compile Model")
model.compile(**compile_model_info)
print()


# ## 5. Training

# from train_segdisclas_utils import *
start_train_info(train_session = train_session, print = print)

train_session["copy_files"] = []
for pattern in params["app_bak"]: train_session["copy_files"].extend(glob.glob(pattern))
train_session["copy_files"].append(params["app_cfg"])
print(train_session["copy_files"])
copy_train_files(copy_files = train_session["copy_files"], train_session = train_session, params = params, print = print)


if params["verbose"] == 2: raise Exception("Exit Training", "Training Session")


####################
## TRAINING
####################   
if params["use_parallel_model"] == True and len(params["gpus"])>=2:
    model = multi_gpu_model(model, gpus = len(params["gpus"]), cpu_merge=False)
    model.compile(**compile_model_info)
# if

model.fit(x = train_gen, 
              initial_epoch   = params["initial_epoch"],
              epochs          = params["final_epoch"],
              steps_per_epoch = params["steps_per_epoch"],

              validation_data = valid_gen,

              validation_steps= params["validation_steps"],

              callbacks       = callbacks,

              max_queue_size  = params["max_queue_size"], 
              workers         = params["workers"], 
              use_multiprocessing = params["use_multiprocessing"],

              verbose         = 1)
####################


end_train_info(train_session = train_session, print = print)


# ## 6. Finishing

copy_train_files(copy_files = train_session["copy_files"], train_session = train_session, params = params, print = print)


# from train_segdisclas_utils import *
dump_train_info(params=params, app_cfg=app_cfg, train_session=train_session, verbose = 0, print = print)


print = print_logs["stop"]()
_is_training_finished = True
del model
del callbacks
gc.collect()
# if params["app_log"] is not None: print = log_print_fns["stop"]()
# IPython.sys.exit(0)


# # END
