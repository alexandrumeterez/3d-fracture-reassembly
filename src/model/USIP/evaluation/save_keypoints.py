import sys
sys.path.append('..')

import os
import time
import copy
import numpy as np
import math
import shutil
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, required=True)
parser.add_argument("--root", type=str, required=True)
parser.add_argument("--detector_model_path", type=str, required=True)
args = parser.parse_args()
print(args)
is_timing = True

dataset_type = 'customnet'
root = args.root
output_folder = os.path.join(root, 'keypoints')

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

assert root is not None
assert output_folder is not None


gpu_id = 0

is_ensure_keypoint_num = True
desired_keypoint_num = 128
NMS_radius = 0.01
noise_sigma = 0
downsample_rate = 1
# =============== method ================
method = 'tsf'
detector_model_path = args.detector_model_path
# detector_model_path = '/data/tsf/oxford/checkpoints/save/detector/BALL-16384-512-r2k64-k16/best.pth'
# detector_model_path = '/data/tsf/scenenn/checkpoints/save/detector/5000-512-k1k32-full-3d/best.pth'
# detector_model_path = '/data/tsf/match3d/checkpoints/save/detector/10240-512-k1k32-full-lambda100/best.pth'
# detector_model_path = '/data/tsf/modelnet/checkpoints/save/no_sigma/lambda_1/5000-512k32-3d/best.pth'

assert detector_model_path is not None, "Please specify the model path"


if dataset_type == 'kitti':
    import kitti.options_detector

    opt_detector_instance = kitti.options_detector.Options()
    opt_detector_instance.parse_without_process()
    opt_detector_instance.opt.gpu_ids = [gpu_id]
    opt_detector_instance.opt.batch_size = 8
    opt_detector_instance.process_opts()
    opt_detector = opt_detector_instance.opt
elif dataset_type == 'oxford':
    import oxford.options_detector

    opt_detector_instance = oxford.options_detector.Options()
    opt_detector_instance.parse_without_process()
    opt_detector_instance.opt.gpu_ids = [gpu_id]
    opt_detector_instance.opt.batch_size = 8
    opt_detector_instance.process_opts()
    opt_detector = opt_detector_instance.opt
elif dataset_type == 'redwood':
    import scenenn.options_detector

    opt_detector_instance = scenenn.options_detector.Options()
    opt_detector_instance.parse_without_process()
    opt_detector_instance.opt.gpu_ids = [gpu_id]
    opt_detector_instance.opt.batch_size = 8
    opt_detector_instance.process_opts()
    opt_detector = opt_detector_instance.opt
elif dataset_type == '3dmatch_eval':
    import match3d.options_detector

    opt_detector_instance = match3d.options_detector.Options()
    opt_detector_instance.parse_without_process()
    opt_detector_instance.opt.gpu_ids = [gpu_id]
    opt_detector_instance.opt.batch_size = 8
    opt_detector_instance.process_opts()
    opt_detector = opt_detector_instance.opt
elif dataset_type == 'modelnet':
    import modelnet.options_detector
    opt_detector = modelnet.options_detector.Options().parse()
elif dataset_type == 'customnet':
    import customnet.options_detector
    opt_detector = customnet.options_detector.Options().parse()
else:
    assert False

opt_detector.input_pc_num = int(opt_detector.input_pc_num / downsample_rate)

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import string

from models.keypoint_detector import ModelDetector
from models.keypoint_descriptor import ModelDescriptor
from evaluation.kitti_test_loader import KittiTestLoader
from evaluation.oxford_test_loader import OxfordTestLoader
from evaluation.redwood_loader import RedwoodLoader
from data.match3d_eval_loader import Match3DEvalLoader
from data.modelnet_rotated_loader import ModelNet_Rotated_Loader
from data.customnet_rotated_loader import CustomNet_Rotated_Loader

def random_string_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))


def model_state_dict_parallel_convert(state_dict, mode):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if mode == 'to_single':
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'to_parallel':
        for k, v in state_dict.items():
            name = 'module.' + k  # add 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'same':
        new_state_dict = state_dict
    else:
        raise Exception('mode = to_single / to_parallel')

    return new_state_dict


def model_state_dict_convert_auto(state_dict, gpu_ids):
    for k, v in state_dict.items():
        if (k[0:7] == 'module.' and len(gpu_ids) >= 2) or (k[0:7] != 'module.' and len(gpu_ids) == 1):
            return state_dict
        elif k[0:7] == 'module.' and len(gpu_ids) == 1:
            return model_state_dict_parallel_convert(state_dict, mode='to_single')
        elif k[0:7] != 'module.' and len(gpu_ids) >= 2:
            return model_state_dict_parallel_convert(state_dict, mode='to_parallel')
        else:
            raise Exception('Error in model_state_dict_convert_auto')


def nms(keypoints_np, sigmas_np, NMS_radius):
    '''

    :param keypoints_np: Mx3
    :param sigmas_np: M
    :return: valid_keypoints_np, valid_sigmas_np, valid_descriptors_np
    '''
    if NMS_radius < 0.01:
        return keypoints_np, sigmas_np

    valid_keypoint_counter = 0
    valid_keypoints_np = np.zeros(keypoints_np.shape, dtype=keypoints_np.dtype)
    valid_sigmas_np = np.zeros(sigmas_np.shape, dtype=sigmas_np.dtype)

    while keypoints_np.shape[0] > 0:
        # print(sigmas_np.shape)
        # print(sigmas_np)

        min_idx = np.argmin(sigmas_np, axis=0)
        # print(min_idx)

        valid_keypoints_np[valid_keypoint_counter, :] = keypoints_np[min_idx, :]
        valid_sigmas_np[valid_keypoint_counter] = sigmas_np[min_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = np.linalg.norm(
            (valid_keypoints_np[valid_keypoint_counter:valid_keypoint_counter + 1, :] - keypoints_np), axis=1,
            keepdims=False)  # M
        mask = distance_array > NMS_radius  # M

        keypoints_np = keypoints_np[mask, ...]
        sigmas_np = sigmas_np[mask]

        # increase counter
        valid_keypoint_counter += 1

    return valid_keypoints_np[0:valid_keypoint_counter, :], \
           valid_sigmas_np[0:valid_keypoint_counter]


def ensure_keypoint_number(frame_keypoint_np, frame_pc_np, keypoint_num):
    if frame_keypoint_np.shape[0] == keypoint_num:
        return frame_keypoint_np
    elif frame_keypoint_np.shape[0] > keypoint_num:
        return frame_keypoint_np[np.random.choice(frame_keypoint_np.shape[0], keypoint_num, replace=False), :]
    else:
        additional_frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], keypoint_num-frame_keypoint_np.shape[0], replace=False), :]
        frame_keypoint_np = np.concatenate((frame_keypoint_np, additional_frame_keypoint_np), axis=0)
        return frame_keypoint_np

if __name__ == '__main__':
    output_folder_real = output_folder + '/' + method
    # augment output_folder with random characters to avoid multi-process inference.
    output_folder = output_folder_real + "_" + random_string_generator(8)

    if 'kitti' == dataset_type:
        testset = KittiTestLoader(test_txt_folder, numpy_folder, opt_detector)
    elif 'oxford' == dataset_type:
        testset = OxfordTestLoader(root_folder, opt_detector)
    elif 'redwood' == dataset_type:
        testset = RedwoodLoader(npy_folder, opt_detector)
    elif '3dmatch_eval' == dataset_type:
        testset = Match3DEvalLoader(npy_folder, opt_detector)
    elif 'modelnet' == dataset_type:
        testset = ModelNet_Rotated_Loader(root, opt_detector)
    elif 'customnet' == dataset_type:
        testset = CustomNet_Rotated_Loader(root, opt_detector)
    else:
        assert False

    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
                                             shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)

    if method == 'tsf':
        # build detector
        model_detector = ModelDetector(opt_detector)
        model_detector.detector.load_state_dict(
            model_state_dict_convert_auto(
                torch.load(
                    detector_model_path,
                    map_location='cpu'), opt_detector.gpu_ids))
        model_detector.freeze_model()


    keypoint_num_list = []
    for i, data in enumerate(testloader):
        if 'kitti' == dataset_type:
            anc_pc, anc_sn, anc_node, seq, anc_idx = data
        elif 'oxford' == dataset_type:
            anc_pc, anc_sn, anc_node, anc_idx = data
        elif 'redwood' == dataset_type or '3dmatch_eval' == dataset_type:
            anc_pc, anc_sn, anc_node, scene_idx, frame_idx = data
        elif 'modelnet' == dataset_type or 'customnet' == dataset_type:
            anc_pc, anc_sn, anc_node, anc_idx, anc_type = data
        else:
            assert False

        # add noise on anc_pc
        anc_pc = anc_pc + torch.randn(anc_pc.size()) * noise_sigma

        # timing
        begin_t = time.time()
        if method == 'tsf':
            anc_pc_cuda = anc_pc.to(opt_detector.device)
            anc_sn_cuda = anc_sn.to(opt_detector.device)
            anc_node_cuda = anc_node.to(opt_detector.device)

            # run detection
            # Bx3xM, BxM
            anc_keypoints, anc_sigmas = model_detector.run_model(anc_pc_cuda, anc_sn_cuda, anc_node_cuda)
            anc_keypoints_np = anc_keypoints.detach().permute(0, 2, 1).contiguous().cpu().numpy()  # BxMx3
            anc_sigmas_np = anc_sigmas.detach().cpu().numpy()  # BxM
        
        elif method == 'random':
            anc_keypoints_list = []
            for b in range(anc_pc.size(0)):
                frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
                frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], desired_keypoint_num, replace=False), :]
                anc_keypoints_list.append(frame_keypoint_np)

        if is_timing:
            print("time consumed per %d frame: %f" % (anc_pc.size(0), time.time()-begin_t))

        for b in range(anc_pc.size(0)):
            frame_pc_np = np.transpose(anc_pc[b].detach().numpy())  # Nx3
            if method == 'tsf':
                frame_keypoint_np = anc_keypoints_np[b]
                frame_sigma_np = anc_sigmas_np[b]

                # nms
                frame_keypoint_np, frame_sigma_np = nms(frame_keypoint_np, frame_sigma_np, NMS_radius=NMS_radius)

                # remove small sigma
                if is_ensure_keypoint_num:
                    sorted_sigma_idx = np.argsort(frame_sigma_np)
                    if desired_keypoint_num > frame_keypoint_np.shape[0]:
                        desired_keypoint_num = frame_keypoint_np.shape[0]
                    sorted_sigma_idx = sorted_sigma_idx[0:desired_keypoint_num]
                    frame_keypoint_np = frame_keypoint_np[sorted_sigma_idx, ...]  # Mx3
                    frame_sigma_np = frame_sigma_np[sorted_sigma_idx]  # M

            else:
                frame_keypoint_np = anc_keypoints_list[b]
                # assure at least one keypoint, by randomly selecting a point
                if frame_keypoint_np.shape[0] == 0:
                    frame_keypoint_np = frame_pc_np[0:1, :]

            # write to file
            if 'kitti' == dataset_type:
                if not os.path.isdir(os.path.join(output_folder, '%02d' % seq[b].item())):
                    os.makedirs(os.path.join(output_folder, '%02d' % seq[b].item()))
                output_file = os.path.join(output_folder, '%02d' % seq[b].item(), '%06d.bin' % anc_idx[b].item())
            elif 'oxford' == dataset_type:
                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)
                output_file = os.path.join(output_folder, '%d.bin' % anc_idx[b].item())
            elif 'redwood' == dataset_type or '3dmatch_eval' == dataset_type:
                if not os.path.isdir(os.path.join(output_folder, testset.scene_name_list[scene_idx[b].item()])):
                    os.makedirs(os.path.join(output_folder, testset.scene_name_list[scene_idx[b].item()]))
                output_file = os.path.join(output_folder, testset.scene_name_list[scene_idx[b].item()], '%d.bin' % frame_idx[b].item())
            elif 'modelnet' == dataset_type:
                if anc_type[b].item() == 0:
                    output_folder_new = os.path.join(output_folder, 'original')
                elif anc_type[b].item() == 1:
                    output_folder_new = os.path.join(output_folder, 'rotated')
                else:
                    assert False
                if not os.path.isdir(output_folder_new):
                    os.makedirs(output_folder_new)
                output_file = os.path.join(output_folder_new, '%d.bin' % anc_idx[b].item())
            elif 'customnet' == dataset_type:
                output_folder_new = output_folder
                if not os.path.isdir(output_folder_new):
                    os.makedirs(output_folder_new)
                output_file = os.path.join(output_folder_new, '%d.bin' % anc_idx[b].item())
                output_file_npy = os.path.join(output_folder_new, '%d.npy' % anc_idx[b].item())
            else:
                assert False
            output = frame_keypoint_np
            output = output.astype(np.float32)
            output.tofile(output_file)
            np.save(output_file_npy, output)
            # print info
            print(output_file + ': %d' % output.shape[0])
            keypoint_num_list.append(output.shape[0])

    keypoint_num_np = np.asarray(keypoint_num_list)
    print('keypoint number max: %d, min: %d, mean: %d' % (np.max(keypoint_num_np),
                                                          np.min(keypoint_num_np),
                                                          round(np.mean(keypoint_num_np))))
    output_folder_real = output_folder_real + '_%d' % (round(np.mean(keypoint_num_np)))
    # output_folder_real = output_folder_real + '_%d_sigma_%.3f' % (round(np.mean(keypoint_num_np)), noise_sigma)
    # output_folder_real = output_folder_real + '_%d_lambda_%.3f' % (round(np.mean(keypoint_num_np)), opt_detector.keypoint_on_pc_alpha)
    # output_folder_real = output_folder_real + '_%d_downsample_%d' % (round(np.mean(keypoint_num_np)), downsample_rate)
    print(output_folder_real)
    try:
        assert False
        shutil.copytree(output_folder, output_folder_real)
        shutil.rmtree(output_folder)
    except Exception as e:
        print(e)
        print("tmp folder is: %s" % output_folder)

