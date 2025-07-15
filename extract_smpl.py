import glob
import os
import sys
import pdb
import os.path as osp
import argparse

sys.path.append(os.getcwd())

# import open3d as o3d
# import open3d.visualization.rendering as rendering
import imageio
from tqdm import tqdm
import joblib
import numpy as np
import torch

from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import random

from smpl_sim.smpllib.smpl_mujoco_new import SMPL_BONE_ORDER_NAMES as joint_names
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from pathlib import Path


def main(params):    
    data_dir = "data/smpl"
    smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
    smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
    smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
    loaded_motion = torch.load(params.record_path)
    print(loaded_motion.keys())

    #pkl_data = joblib.load(params.record_path)
    mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]
    
    len_items = loaded_motion["global_rotation"].shape[0]

    npy_dict = {
        'motion': [],
        'text': [],
        'lengths': [],
        'num_samples': len_items,
        'num_repetitions': 1,
    }
    
    # skeleton_tree = SkeletonTree.from_mjcf("/tmp/smpl/smpl_humanoid_acc69a33-e2dd-4737-ac8d-5af97df13d00.xml")

    skeleton_tree = SkeletonTree.from_mjcf("./protomotions/data/assets/mjcf/smpl_humanoid.xml")
    # template_file = "../PHC/output/states/phc_comp_kp_2-2025-04-09-20:14:07.pkl"
    
    preloaded_skeleton_tree = skeleton_tree #joblib.load(template_file)["0_0"]["skeleton_tree"]
    
    reset_inds = torch.where(loaded_motion["is_done"] == 1)[0].tolist()
    assert params.num_resets > 0, "num_resets must be greater than 0"
    assert params.num_resets < len(reset_inds), "num_resets must be less than the number of resets in the motion"
    print("reset_inds", reset_inds)
    
    for seg_i in range(params.num_resets):
        start_frame, end_frame = reset_inds[seg_i], reset_inds[seg_i+1]

        pose_quat, trans = loaded_motion['global_rotation'].cpu().numpy(), loaded_motion['global_translation'][:, 0, :].cpu().numpy()
        
        pose_quat = pose_quat[start_frame:end_frame]
        trans = trans[start_frame:end_frame]
        
        skeleton_tree = preloaded_skeleton_tree
        #skeleton_tree = SkeletonTree.from_dict(preloaded_skeleton_tree)
        offset = skeleton_tree.local_translation[0]

        root_trans_offset = trans - offset.numpy()
        root_trans_offset = root_trans_offset - root_trans_offset[0:1]  # now root_trans_offset[0] == 0
        print(root_trans_offset[..., -1], root_trans_offset[0])

        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=False)

        global_rot = sk_state.global_rotation
        
        B, J, N = global_rot.shape
        pose_quat = (sRot.from_quat(global_rot.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5])).as_quat().reshape(B, -1, 4)
        B_down = pose_quat.shape[0]
        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=False)
        local_rot = new_sk_state.local_rotation
	
	
        pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_matrix()[:, :2, :].reshape(B_down, -1, 6)  # 6d representation
        pose_aa = pose_aa[:, mujoco_2_smpl, :]
        
        roop_pose_pad = np.concatenate([root_trans_offset, np.zeros_like(root_trans_offset)], axis=-1)[:, None]
        motion_rep = np.concatenate([pose_aa, roop_pose_pad], axis=1)[None].transpose(0, 2, 3, 1)
        
        npy_dict['motion'].append(motion_rep)
        npy_dict['lengths'].append(int(motion_rep.shape[-1]))
        npy_dict['text'].append("")
        
    npy_dict['lengths'] = np.array(npy_dict['lengths'])
    _moption = np.zeros([len(npy_dict['lengths']), 25, 6, npy_dict['lengths'].max()], dtype=npy_dict['motion'][0].dtype)
    for i, _len in enumerate(npy_dict['lengths']):
        _moption[i, :, :, :_len] = npy_dict['motion'][i]
    npy_dict['motion'] = _moption
    print("Motion shape:", npy_dict['motion'].shape)

    out_path = params.record_path.replace('.pkl', '_smpl.npy')
    np.save(out_path, npy_dict)
    np.save( str(Path(out_path).parent) + "/" + params.save_name + ".npy", npy_dict)
    print(f'Saved [{os.path.abspath(out_path)}]')
    print( "and " , str(Path(out_path).parent) + "/" + params.save_name + ".npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_path", type=str, required=True, help='Path to pkl state file')
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--num_resets", type=int, default=1, help='Number of resets to process')
    params = parser.parse_args()
    main(params)