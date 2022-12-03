import copy
from stable_diffusion_quantizer.mapper import *
from stable_diffusion_quantizer.unet.calibrator import stable_diffusion_quantizer_unet_calibrate
from stable_diffusion_quantizer.unet.qat import stable_diffusion_quantizer_unet_qat
# dataset
from stable_diffusion_quantizer.dataset_utils import download_webdataset
from stable_diffusion_quantizer.dataset import DataPipelineTTI
import logging


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def stable_diffusion_quantizer_unet(pipeline, cfg, *args, **kwargs):
    logger.debug("start UNet quantization...")
    # download pipeline from GPU
    device = pipeline.device
    pipeline.to("cpu")
    # copy models
    logger.debug("initilize auxiliary pipeline...")
    _pipeline = copy.deepcopy(pipeline).to(cfg.model.device)
    unet_teacher = copy.deepcopy(pipeline.unet)
    # make post-training quantization
    logger.debug("initilize quantized layers...")
    names = get_names(_pipeline.unet, **cfg.calibrator)
    if cfg.calibrator.steps > 0:
        logger.debug("calibration...")
        modules_to_calibrator(_pipeline.unet, names, **cfg.calibrator)
        _pipeline.to(cfg.calibrator.device)
        stable_diffusion_quantizer_unet_calibrate(_pipeline, **cfg.calibrator)
        _pipeline.to("cpu")
        calibrator_to_qmodule(_pipeline.unet, names, **cfg.calibrator)
    else:
        modules_to_qmodules(_pipeline.unet, names, **cfg.calibrator)
    # make quantization aware training
    logger.debug("download dataset...")
    trainset_cfg = download_webdataset(**cfg.trainset)
    logger.debug("load dataset...")
    trainset = DataPipelineTTI(
        data_path=trainset_cfg.path,
        tokenizer=pipeline.tokenizer
    )
    logger.debug("quantization-aware training...")
    _pipeline = stable_diffusion_quantizer_unet_qat(
        pipeline=_pipeline,
        dataset=trainset,
        unet_teacher=unet_teacher,
        **cfg.qat
    )
    logger.debug("finalize UNet quantization...")
    pipeline.unet = _pipeline.unet
    pipeline.to(device)
    return pipeline


