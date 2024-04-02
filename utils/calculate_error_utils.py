import numpy as np

#The rotation error is defined as the angle required to rotate from the rotation matrix cur_R to the rotation matrix obs_R.
def cal_rot_error(cur_R,obs_R):

    trace = np.trace(np.dot(cur_R, np.linalg.inv(obs_R)))
    trace = max(-1, min(3, trace))
    angle_diff = np.arccos(0.5 * (trace - 1))
    angle_diff_deg = np.rad2deg(angle_diff)
    
    return angle_diff_deg

#The translation error is defined as the Euclidean distance from the translation vector cur_T to the translation vector obs_T.
def cal_tran_error(cur_T,obs_T):
    
    return np.linalg.norm(cur_T-obs_T)

#Calculate the rotation and translation errors between two camera poses.
def cal_campose_error(cur_pose_c2w,obs_pose_c2w):
 
    rot_error=cal_rot_error(cur_pose_c2w[:3,:3],obs_pose_c2w[:3,:3])
    translation_error = cal_tran_error(cur_pose_c2w[:3,3],obs_pose_c2w[:3,3])

    return rot_error,translation_error