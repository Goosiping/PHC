import glob
import os
import sys
import pdb
import os.path as osp
import scipy.ndimage.filters as filters

sys.path.append(os.getcwd())

import torch 
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
import argparse
import cv2
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

from poselib.poselib.visualization.common import plot_skeleton_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="")
    args = parser.parse_args()
    
    process_split = "train"
    upright_start = True
    axis_transform = True
    robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": upright_start,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True, 
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False, 
            "box_body": False,
            "master_range": 50,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smpl",
        }

    smpl_local_robot = LocalRobot(robot_cfg,)
    # if not osp.isdir(args.path):
    #     print("Please specify AMASS data path")
        
    # all_pkls = glob.glob(f"{args.path}/**/*.npz", recursive=True)
    # amass_occlusion = joblib.load("sample_data/amass_copycat_occlusion_v3.pkl")
    # amass_splits = {
    #     'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    #     'test': ['Transitions_mocap', 'SSM_synced'],
    #     'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT',  'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset", "DFaust_67"]   # Adding ACCAD
    # }
    # process_set = amass_splits[process_split]
    length_acc = []
    amass_full_motion_dict = {}

    # Start processing one file
    data_path = "./data/table_tennis/res.pk"
    
    # for data_path in tqdm(all_pkls):
    bound = 0
    splits = data_path.split("/")[7:]
    key_name_dump = "0-" + "_".join(splits).replace(".npz", "")
    
        
    # entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    entry_data = joblib.load(data_path)
    
    framerate = 30

    if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
        bound = -2
    
    # rotation matrix transition
    T = entry_data['pred_thetas'].shape[0]
    J = 24
    rot_mat = entry_data['pred_thetas'].reshape(-1, 3, 3)
    pose_aa = sRot.from_matrix(rot_mat).as_rotvec()
    pose_aa = pose_aa.reshape(T, J * 3)

    if axis_transform:
        transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False)
        new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
        pose_aa[:, :3] = new_root

    skip = int(framerate/30)
    root_trans = entry_data['transl'][::skip, :].squeeze()
    # pose_aa = np.concatenate([entry_data['pose_24'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['pred_betas'][0]
    gender = entry_data.get("gender", "neutral")
    N = pose_aa.shape[0]
    
    if bound == 0:
        bound = N
        
    root_trans = root_trans[:bound]
    pose_aa = pose_aa[:bound]
    N = pose_aa.shape[0]
    # root_trans[:,2] = 1
    root_trans[:, [1, 2]] = root_trans[:, [2, 1]]

    # Guassian for root trans
    # filtered_root_trans = np.array(root_trans)
    # filtered_root_trans[..., 2] = filters.gaussian_filter1d(filtered_root_trans[..., 2], 10, axis=0, mode="mirror") # More filtering on the root translation
    # filtered_root_trans[..., :2] = filters.gaussian_filter1d(filtered_root_trans[..., :2], 5, axis=0, mode="mirror")
    # root_trans = filtered_root_trans

    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

    beta = np.zeros((16))
    gender_number, beta[:], gender = [0], 0, "neutral"
    # print("using neutral model")
    smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
    smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                torch.from_numpy(pose_quat),
                root_trans_offset,
                is_local=True)
    
    # try to visualize
    # from ipdb import set_trace;set_trace()
    # plot_skeleton_state(new_sk_state)
    
    if robot_cfg['upright_start'] or axis_transform:
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
        pose_quat = new_sk_state.local_rotation.numpy()


    pose_quat_global = new_sk_state.global_rotation.numpy()
    pose_quat = new_sk_state.local_rotation.numpy()
    fps = 30

    # Try to touch floor
    # import ipdb;ipdb.set_trace()
    # kpts = entry_data['pred_pose']
    # print("INFO: kpts shape ", kpts.shape)
    # kpts_y = kpts.view(T, 48, 3)[:,:,1]
    # per_frame_offset = kpts_y.min(dim=1, keepdim=True)[0].abs()
    # print("INFO: per_frame_offset shape ", per_frame_offset.shape)
    # root_trans[:,1] = 10
    # root_trans_offset[:,2] = 0

    new_motion_out = {}
    new_motion_out['pose_quat_global'] = pose_quat_global
    new_motion_out['pose_quat'] = pose_quat
    new_motion_out['trans_orig'] = root_trans
    new_motion_out['root_trans_offset'] = root_trans_offset
    new_motion_out['beta'] = beta
    new_motion_out['gender'] = gender
    new_motion_out['pose_aa'] = pose_aa
    new_motion_out['fps'] = fps

    amass_full_motion_dict[key_name_dump] = new_motion_out
    # end of for loop
        
    # import ipdb; ipdb.set_trace()
    # output_dir = "data/table_tennis/converted_upright_axis_filter.pkl"
    output_dir = "data/table_tennis/converted_hybrik_trans2.pkl"
    # output_dir = "data/tram_example.pkl"
    if upright_start:
        joblib.dump(amass_full_motion_dict, output_dir, compress=True)
    else:
        joblib.dump(amass_full_motion_dict, output_dir, compress=True)
    print("INFO: Finished")