import os
from infer import *
import paddle

class model(object):
    def __init__(self,model_dir,
        output_dir,
        device="cpu",
        run_mode="paddle",
        batch_size=1,
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        trt_calib_mode=False,
        cpu_threads=1,
        enable_mkldnn=False,
        enable_mkldnn_bfloat16=False,
        threshold=0.5,
    ):
        self.deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        paddle.enable_static()
        with open(self.deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        arch = yml_conf['arch']
        detector_func = 'Detector'
        if arch == 'SOLOv2':
            detector_func = 'DetectorSOLOv2'
        elif arch == 'PicoDet':
            detector_func = 'DetectorPicoDet'
        self.detector= eval(detector_func)(
            model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
            threshold=threshold,
            output_dir=output_dir
        )
    
    def predict_img(self,image_file,image_dir=None):
        """
        预测图片
        image_file : 图片路径
        """
        img_list = get_test_images(image_dir, image_file,status=0)
        results,box_list=self.detector.predict_image(
            img_list, False, repeats=100, save_file=None)
        print(box_list)

models =model("/media/cyl/BA9D-BAE0/ppyolo_r50vd_dcn_voc","./")
models.predict_img("/media/cyl/BA9D-BAE0/微信图片_20220611163415.jpg")