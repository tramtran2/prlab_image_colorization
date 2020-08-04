"""
model_name: 
    + seg:
        segdisclasunet
        qubvel_sm_unet
    + segdisclas:
        segdisclasunet

"""
import segmentation_models as sm
from .segclas_colorized_unet import segclas_colorized_unet_v0, segclas_colorized_unet_v0_cfg
from .zhang_models import zhang_vgg16_normal_build
from .zhao_models import zhao_vgg16_normal_build

class FactoryModels(object):
    
    model_type_list = ["soft_colorized", "reg_colorized", "regsoft_colorized", "clasregsoft_colorized", "segclasregsoft_colorized"]
    
    @classmethod
    def create(cls, model_type, model_name, model_cfg, **kwargs):
        result = None
        if model_type=="soft_colorized":
            result = cls.create_soft_colorized(model_name, model_cfg, **kwargs)
        elif model_type=="reg_colorized":
            result = cls.create_reg_colorized(model_name, model_cfg, **kwargs)
        elif model_type=="regsoft_colorized":
            result = cls.create_regsoft_colorized(model_name, model_cfg, **kwargs)
        elif model_type=="clasregsoft_colorized":
            result = cls.create_clasregsoft_colorized(model_name, model_cfg, **kwargs)            
        elif model_type=="segclasregsoft_colorized":
            result = cls.create_segclasregsoft_colorized(model_name, model_cfg, **kwargs)            
        elif model_type=="zhang_vgg16":
            result = cls.create_zhang_vgg16(model_name, model_cfg, **kwargs)                        
        elif model_type=="zhao_vgg16":
            result = cls.create_zhao_vgg16(model_name, model_cfg, **kwargs)                 
        # if
        return result
        pass
    # create
    
    @classmethod
    def create_soft_colorized(cls, model_name, model_cfg, params, **kwargs):
        """
        Input: 
        Output: 
        """
        result = {"model": None, "preprocess_input": None}
        
        if model_name=="segclas_colorized_unet_v0":
            model_info = segclas_colorized_unet_v0_cfg["soft_colorized"]
            model_info.update(**model_cfg)

            result["model"] = segclas_colorized_unet_v0(**model_info)
            pass
        # if
        return result
    # create_se_colorized

    @classmethod
    def create_reg_colorized(cls, model_name, model_cfg, params, **kwargs):
        """
        Input: 
        Output: 
        """
        result = {"model": None, "preprocess_input": None}
       
        if model_name=="segclas_colorized_unet_v0":
            model_info = segclas_colorized_unet_v0_cfg["reg_colorized"]
            model_info.update(**model_cfg)

            result["model"] = segclas_colorized_unet_v0(**model_info)
            pass
        # if
        return result
    # create_reg_colorized    

    @classmethod
    def create_regsoft_colorized(cls, model_name, model_cfg, params, **kwargs):
        """
        Input: 
        Output: 
        """
        result = {"model": None, "preprocess_input": None}
       
        if model_name=="segclas_colorized_unet_v0":
            model_info = segclas_colorized_unet_v0_cfg["regsoft_colorized"]
            model_info.update(**model_cfg)

            result["model"] = segclas_colorized_unet_v0(**model_info)
            pass
        # if
        return result
    # create_regsoft_colorized 

    @classmethod
    def create_clasregsoft_colorized(cls, model_name, model_cfg, params, **kwargs):
        """
        Input: 
        Output: 
        """
        result = {"model": None, "preprocess_input": None}
       
        if model_name=="segclas_colorized_unet_v0":
            model_info = segclas_colorized_unet_v0_cfg["clasregsoft_colorized"]
            model_info.update(**model_cfg)

            result["model"] = segclas_colorized_unet_v0(**model_info)
            pass
        # if
        return result
    # create_clasregsoft_colorized 

    @classmethod
    def create_segclasregsoft_colorized(cls, model_name, model_cfg, params, **kwargs):
        """
        Input: 
        Output: 
        """
        result = {"model": None, "preprocess_input": None}
       
        if model_name=="segclas_colorized_unet_v0":
            model_info = segclas_colorized_unet_v0_cfg["segclasregsoft_colorized"]
            model_info.update(**model_cfg)

            result["model"] = segclas_colorized_unet_v0(**model_info)
            pass
        # if
        return result
    # create_segclasregsoft_colorized 

    @classmethod
    def create_zhang_vgg16(cls, model_name, model_cfg, params, **kwargs):
        """
        Input: 
        Output: 
        """
        result = {"model": None, "preprocess_input": None}
        result["model"] = zhang_vgg16_normal_build(**model_cfg)
        return result
    # create_zhang_vgg16     

    @classmethod
    def create_zhao_vgg16(cls, model_name, model_cfg, params, **kwargs):
        """
        Input: 
        Output: 
        """
        result = {"model": None, "preprocess_input": None}
        result["model"] = zhao_vgg16_normal_build(**model_cfg)
        return result
    # create_zhang_vgg16    
# FactoryModels

