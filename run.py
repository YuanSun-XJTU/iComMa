import torch
from scene import Scene
import torch.optim as optim
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.calculate_error_utils import cal_campose_error
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.icomma_helper import load_LoFTR, get_pose_estimation_input
from utils.image_utils import to8b
import cv2
import imageio
import os
import ast
from scene.cameras import Camera_Pose
from utils.loss_utils import loss_loftr,loss_mse

def camera_pose_estimation(gaussians:GaussianModel, background:torch.tensor, pipeline:PipelineParams, icommaparams:iComMaParams, icomma_info, output_path):
    # start pose & gt pose
    gt_pose_c2w=icomma_info.gt_pose_c2w
    start_pose_w2c=icomma_info.start_pose_w2c.cuda()
    
    # query_image for comparing 
    query_image = icomma_info.query_image.cuda()

    # initialize camera pose object
    camera_pose = Camera_Pose(start_pose_w2c,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_pose.cuda()

    # store gif elements
    imgs=[]
    
    matching_flag= not icommaparams.deprecate_matching

    # start optimizing
    optimizer = optim.Adam(camera_pose.parameters(),lr = icommaparams.camera_pose_lr)
    iter = icommaparams.pose_estimation_iter
    num_iter_matching = 0
    for k in range(iter):

        rendering = render(camera_pose,gaussians, pipeline, background,compute_grad_cov2d = icommaparams.compute_grad_cov2d)["render"]

        if matching_flag:
            loss_matching = loss_loftr(query_image,rendering,LoFTR_model,icommaparams.confidence_threshold_LoFTR,icommaparams.min_matching_points)
            loss_comparing = loss_mse(rendering,query_image)
            
            if loss_matching is None:
                loss = loss_comparing
            else:  
                loss = icommaparams.lambda_LoFTR *loss_matching + (1-icommaparams.lambda_LoFTR)*loss_comparing
                if loss_matching<0.001:
                    matching_flag=False
                    
            num_iter_matching += 1
        else:
            loss_comparing = loss_mse(rendering,query_image)
            loss = loss_comparing
            
            new_lrate = icommaparams.camera_pose_lr * (0.6 ** ((k - num_iter_matching + 1) / 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        
        # output intermediate results
        if (k + 1) % 20 == 0 or k == 0:
            print('Step: ', k)
            if matching_flag and loss_matching is not None:
                print('Matching Loss: ', loss_matching.item())
            print('Comparing Loss: ', loss_comparing.item())
            print('Loss: ', loss.item())

            # record error
            with torch.no_grad():
                cur_pose_c2w= camera_pose.current_campose_c2w()
                rot_error,translation_error=cal_campose_error(cur_pose_c2w,gt_pose_c2w)
                print('Rotation error: ', rot_error)
                print('Translation error: ', translation_error)
                print('-----------------------------------')
               
            # output images
            if icommaparams.OVERLAY is True:
                with torch.no_grad():
                    rgb = rendering.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename = os.path.join(output_path, str(k)+'.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        camera_pose(start_pose_w2c)

    # output gif
    if icommaparams.OVERLAY is True:
        imageio.mimwrite(os.path.join(output_path, 'video.gif'), imgs, fps=4)
  
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Camera pose estimation parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    icommaparams = iComMaParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", default='output', type=str,help="output path")
    parser.add_argument("--obs_img_index", default=0, type=int)
    parser.add_argument("--delta", default="[30,10,10,0.1,0.1,0.1]", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
   
    # Initialize system state (RNG)
    safe_state(args.quiet)

    makedirs(args.output_path, exist_ok=True)
    
    # load LoFTR_model
    LoFTR_model=load_LoFTR(icommaparams.LoFTR_ckpt_path,icommaparams.LoFTR_temp_bug_fix)
    
    # load gaussians
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # get camera info from Scene
    # Reused 3DGS code to obtain camera information. 
    # You can customize the iComMa_input_info in practical applications.
    scene = Scene(dataset,gaussians,load_iteration=args.iteration,shuffle=False)
    obs_view=scene.getTestCameras()[args.obs_img_index]
    #obs_view=scene.getTrainCameras()[args.obs_img_index]
    icomma_info=get_pose_estimation_input(obs_view,ast.literal_eval(args.delta))
    
    # pose estimation
    camera_pose_estimation(gaussians,background,pipeline,icommaparams,icomma_info,args.output_path)