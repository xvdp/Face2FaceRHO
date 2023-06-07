"""@xvdp
simplify script test

e.g.
# S='/home/data/Proto/face_crop_512/face_crop_from_exr_512/ffhq_1365_8b.png'
# D='/home/data/Proto/face_crop_512/face_crop_from_ffhq/44260.png'
# python src/fitting.py --device cuda --src_img $S --drv_img $D --output_src_headpose "./SHeadpose.txt" --output_src_landmark "./SLmk.txt" --output_drv_headpose "./DHeadpose.txt" --output_drv_landmark "./DLmk.txt"
# python src/reenact.py --config ./src/config/test_face2facerho.ini --src_img $S --src_headpose "./SHeadpose.txt" --src_landmark "./SLmk.txt" --drv_headpose "./DLmk.txt" --drv_landmark "./DHeadpose.txt" --output_dir '.'



python src/fitting.py 
--device cuda 
--src_img $S 
--drv_img $D 
--output_src_headpose "./SHeadpose.txt" 
--output_src_landmark "./SLmk.txt" 
--output_drv_headpose "./DHeadpose.txt" 
--output_drv_landmark "./DLmk.txt"

python src/reenact.py
--config ./src/config/test_face2facerho.ini
--src_img $S
--src_headpose "./SHeadpose.txt"
--src_landmark "./SLmk.txt"
--drv_headpose "./DLmk.txt"
--drv_landmark "./DHeadpose.txt"
--output_dir '.'


python src/scripts.py --src_img $s --drv_img $d --out_dir $odir --out_name `basename $s`

"""
from typing import List, Union
import os
import os.path as osp
import argparse
import torch
import cv2

from options.parse_config import Face2FaceRHOConfigParse
from models import create_model
from util.util import save_coeffs, save_landmarks, tensor2im
from util.landmark_image_generation import LandmarkImageGeneration
from fitting import FLAMEFitting, PoseLandmarkExtractor
from reenact import load_data


# pylint: disable=no-member

def parse_args():
    """Configurations."""
    parser = argparse.ArgumentParser(description='test process of Face2FaceRHO')
    parser.add_argument('--src_img', type=str, required=True,
                        help='input source actor image (.jpg, .jpg, .jpeg, .png)')
    parser.add_argument('--drv_img', type=str, required=True,
                        help='driving image (.jpg, .jpg, .jpeg, .png)')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'demo_out'),
                        help='output directory')
    parser.add_argument('--out_name', type=str, default="f2f_result.png", help='output filename')
    return parser.parse_args()

def fit_reenact(src_img: str,
                drv_img: str,
                out_dir: str = 'demo_out',
                src_prefix: str = 'src',
                drv_prefix: str = 'drv',
                device: Union[str, torch.device, None] = None,
                overwrite: bool = True,
                config: str = "src/config/test_face2facerho.ini",
                out_name: str = "f2f_result.png")-> str:
    """ shortcut to fit and reenact demo scripts in read
    needs to be run from FACE2FACERHO/
    Args
        src_img
        drv_img
    optional
        out_dir
        src_prefix # prefixes of extracted temp transforms
        drv_prefix  
        device      cuda if found
        overwrite   True
        config
        out_name    abspath(expanduser(out_dir/out_name)
    Example
        fit_reenac()
    """
    # src_pose_name, src_lmk_name, drv_pose_name, drv_lmk_name]
    poselmks = fit(src_img, drv_img, out_dir, src_prefix, drv_prefix, device, overwrite)
    return reenact(src_img, *poselmks, out_dir, config, out_name)


def reenact(img: str,
            src_pose: str,
            src_lmks: str,
            drv_pose: str,
            drv_lmks: str,
            out_dir: str = 'demo_out',
            config: str = "src/config/test_face2facerho.ini",
            out_name: str = "f2f_result.png") -> str:
    """
    Args
    """
    files = [img, src_pose, src_lmks, drv_pose, drv_lmks]
    for i, file in enumerate(files):
        assert osp.isfile(file), f"file {[i]} not found: {files[i]}"
    out_dir = osp.abspath(osp.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    config_parse = Face2FaceRHOConfigParse()
    opt = config_parse.get_opt_from_ini(config)
    config_parse.setup_environment()

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    src = load_data(opt, src_pose, src_lmks, img)
    drv = load_data(opt, drv_pose, drv_lmks, load_img=False)

    landmark_img_generator = LandmarkImageGeneration(opt)

    # off-line stage
    src['landmark_img'] = landmark_img_generator.generate_landmark_img(src['landmarks'])
    src['landmark_img'] = [value.unsqueeze(0) for value in src['landmark_img']]
    model.set_source_face(src['img'].unsqueeze(0), src['headpose'].unsqueeze(0))

    # on-line stage
    drv['landmark_img'] = landmark_img_generator.generate_landmark_img(drv['landmarks'])
    drv['landmark_img'] = [value.unsqueeze(0) for value in drv['landmark_img']]
    model.reenactment(src['landmark_img'], drv['headpose'].unsqueeze(0), drv['landmark_img'])
    visual_results = model.get_current_visuals()

    out_name = osp.join(out_dir,  out_name)
    if osp.splitext(out_name)[-1].lower() not in ('.png', '.jpg'):
        out_name += ".png"

    out_img = cv2.cvtColor(tensor2im(visual_results['fake']), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_name, out_img)
    return out_name


def fit_flame(img, face_fitting = None, device = None):
    if face_fitting is None:
        device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        face_fitting = FLAMEFitting(device=device)
    return face_fitting.fitting(img)

def flame_to_pose_lmk(params, pose_lml_extractor=None):
    if pose_lml_extractor is None:
        pose_lml_extractor = PoseLandmarkExtractor()

    pose = pose_lml_extractor.get_pose(
        params['shape'], params['exp'], params['pose'],
        params['scale'], params['tx'], params['ty'])

    lmks = pose_lml_extractor.get_project_points(
        params['shape'], params['exp'], params['pose'],
        params['scale'], params['tx'], params['ty'])
    return pose, lmks


def fit(src_img: str,
        drv_img: str,
        out_dir: str = 'demo_out',
        src_prefix: str = 'src',
        drv_prefix: str = 'drv',
        device: Union[str, torch.device, None] = None,
        overwrite: bool = True)-> List[str]:
    """
    Args
        src_img     (str) image path
        drv_img     (str) image_path of driving image
        out_dir     (str ['demo_out']) out_path for stored transform text files
        src_prefix
        drv_prefix
    """
    assert osp.isfile(src_img), f"src not found: {src_img}"
    assert osp.isfile(drv_img), f"dst not found: {drv_img}"
    out_dir = osp.abspath(osp.expanduser(out_dir))
    src_pose_name = osp.join(out_dir, f"{src_prefix}_pose.txt")
    src_lmk_name = osp.join(out_dir, f"{src_prefix}_lmks.txt")
    drv_pose_name = osp.join(out_dir, f"{drv_prefix}_pose.txt")
    drv_lmk_name = osp.join(out_dir, f"{drv_prefix}_lmks.txt")
    out_names = [src_pose_name, src_lmk_name, drv_pose_name, drv_lmk_name]
    if not overwrite:
        for name in out_names:
            assert not osp.isfile(name), f"'{name}' exists, change prefix, out_dir or overwrite arg"
    os.makedirs(out_dir, exist_ok=True)

    device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    face_fitting = FLAMEFitting(device=device)
    src_params = face_fitting.fitting(src_img)
    drv_params = face_fitting.fitting(drv_img)

    pose_lml_extractor = PoseLandmarkExtractor()
    src_headpose = pose_lml_extractor.get_pose(
        src_params['shape'], src_params['exp'], src_params['pose'],
        src_params['scale'], src_params['tx'], src_params['ty'])

    src_lmks = pose_lml_extractor.get_project_points(
        src_params['shape'], src_params['exp'], src_params['pose'],
        src_params['scale'], src_params['tx'], src_params['ty'])

    drv_headpose = pose_lml_extractor.get_pose(
        src_params['shape'], drv_params['exp'], drv_params['pose'],
        drv_params['scale'], drv_params['tx'], drv_params['ty'])

    drv_lmks = pose_lml_extractor.get_project_points(
        src_params['shape'], drv_params['exp'], drv_params['pose'],
        drv_params['scale'], drv_params['tx'], drv_params['ty'])


    save_coeffs(src_pose_name, src_headpose)
    save_landmarks(src_lmk_name, src_lmks)
    save_coeffs(drv_pose_name, drv_headpose)
    save_landmarks(drv_lmk_name, drv_lmks)

    return out_names

if __name__ == '__main__':
    args = parse_args()
    fit_reenact(args.src_img, args.drv_img, args.out_dir, out_name=args.out_name)
