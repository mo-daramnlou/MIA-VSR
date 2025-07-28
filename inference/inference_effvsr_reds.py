import cv2
import glob
import logging
import os
import os.path as osp
import torch
import torch.nn.functional as F
import h5py
from archs.mia_vsr_arch import MIAVSR
from archs.eff_vsr_arch import EFFVSR
from archs.gen_vsr_arch import GENVSR
from archs.bi_vsr_arch import BIVSR
from basicsr.data.data_util import read_img_seq
from basicsr.metrics import psnr_ssim
from basicsr.utils import get_root_logger, get_time_str, imwrite, tensor2img
import time


def main():
    print("Inference_miavsr_reds")
    # -------------------- Configurations -------------------- #
    device = torch.device('cuda:0')
    save_imgs = False
    measure_inference_time = True
    test_y_channel = False
    crop_border = 0
    # set suitable value to make sure cuda not out of memory
    # interval = 30
    
    model_path = '/home/mohammad/Documents/uni/deeplearning/FinalProject/Logs/experiments/ex_genvsr2/content/MIA-VSR/experiments/4131_GENVSR_mix_precision_REDS_600K_N1/models/net_g_60000.pth'
    # test data
    test_name = f'sotareds'

    lr_folder = '/home/mohammad/Documents/uni/deeplearning/FinalProject/inference_data/lr'
    gt_folder = '/home/mohammad/Documents/uni/deeplearning/FinalProject/inference_data/gt'
    save_folder = f'/home/mohammad/Documents/uni/deeplearning/FinalProject/inference_data/results/{test_name}'
    os.makedirs(save_folder, exist_ok=True)

    # logger
    log_file = osp.join(save_folder, f'psnr_ssim_test_{get_time_str()}.log')
    logger = get_root_logger(logger_name='recurrent', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Data: {test_name} - {lr_folder}')
    logger.info(f'Model path: {model_path}')

    # set up the models
    # model = BIVSR()
    model = GENVSR(mid_channels=28, num_blocks=4)
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    if measure_inference_time:
        # lr_folder = '/home/mohammad/Documents/uni/deeplearning/FinalProject/Logs/Inference Time/data/lr'
        # gt_folder = '/home/mohammad/Documents/uni/deeplearning/FinalProject/Logs/Inference Time/data/gt'
        lr_folder = '/home/mohammad/Documents/uni/deeplearning/FinalProject/data/val_sharp_bicubic/val/val_sharp_bicubic/X4'
        gt_folder = '/home/mohammad/Documents/uni/deeplearning/FinalProject/data/val_sharp/val/val_sharp'
        save_folder = f'/home/mohammad/Documents/uni/deeplearning/FinalProject/Logs/Inference Time/data/res{time.time()}'
        # -------------------- Warm-up for stable measurements -------------------- #
        logger.info('Warming up GPU for 10 iterations...')
        # Create a dummy input tensor. The size should be representative of your actual input.
        # Based on your print log: [1, 4, 3, 180, 320]
        # We use a slightly smaller frame count for a quick warm-up.
        dummy_input = torch.randn(1, 4, 3, 180, 320, device=device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        torch.cuda.synchronize()  # Wait for warm-up to finish

    avg_psnr_l = []
    avg_ssim_l = []
    inference_times = []
    num_frames_list = []
    subfolder_l = sorted(glob.glob(osp.join(lr_folder, '*')))
    subfolder_gt_l = sorted(glob.glob(osp.join(gt_folder, '*')))

    print(len(subfolder_l))
    # print("Total Flops",model.flops() / 1e9)

    # for each subfolder
    subfolder_names = []
    for i, (subfolder, subfolder_gt) in enumerate(zip(subfolder_l, subfolder_gt_l)):
        subfolder_name = osp.basename(subfolder)
        subfolder_names.append(subfolder_name)

        # read lq and gt images
        imgs_lq, imgnames = read_img_seq(subfolder, return_imgname=True)
        print(torch.min(imgs_lq))
        print(torch.max(imgs_lq))

        # calculate the iter numbers
        length = len(imgs_lq)
        num_frames_list.append(length)
        # iters = length // interval

        # cluster the excluded file into another group
        # if length % interval > 1:
        #     iters += 1

        avg_psnr = 0
        avg_ssim = 0
        # inference
        name_idx = 0
        imgs_lq = imgs_lq.unsqueeze(0).to(device)
        # for i in range(iters):
        #     min_id = min((i + 1) * interval, length)
        #     lq = imgs_lq[:, i * interval:min_id, :, :, :]
        # print("Inference_miavsr_reds, imgs_lq size:", imgs_lq.size()) #[1, 4, 3, 180, 320]

        if measure_inference_time:
            # --- Timing using torch.cuda.Event ---
            torch.cuda.synchronize(device)  # Wait for all previous GPU work to finish
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            if measure_inference_time:
                start_event.record()

            outputs, _, anchor_feats = model(imgs_lq)

            if  measure_inference_time:
                end_event.record()
                torch.cuda.synchronize(device)  # Wait for the model call to complete
                elapsed_time_ms = start_event.elapsed_time(end_event)
                inference_times.append(elapsed_time_ms)

            outputs = outputs.squeeze(0)
            # print("Inference_miavsr_reds, outputs size:", outputs.size())

            
        # convert to numpy image
        for idx in range(outputs.shape[0]):
            img_name = imgnames[name_idx] + '.png'
            output = tensor2img(outputs[idx], rgb2bgr=True, min_max=(0, 1))
            # read GT image
            img_gt = cv2.imread(osp.join(subfolder_gt, img_name), cv2.IMREAD_UNCHANGED)
            crt_psnr = psnr_ssim.calculate_psnr(
                output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
            crt_ssim = psnr_ssim.calculate_ssim(
            output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
            # save
            if save_imgs:
                s_folder = osp.join(save_folder, subfolder_name, f'{img_name}')
                imwrite(output, s_folder)
                # for key in anchor_feats:
                #     anchor_feat = anchor_feats[key].squeeze()
                #     print("anchor_feat: ",anchor_feat)
                #     anchor_feat = anchor_feat[idx].squeeze()
                    # feat = tensor2img(normalize_tensor(anchor_feat), rgb2bgr=True, min_max=(0,1))
                    # # feat = anchor_feat[idx].squeeze()
                    # print("feat: ",feat)
                    # s_folder = osp.join(save_folder, subfolder_name, f'{imgnames[name_idx]}_{key}' + '.png')
                    # imwrite(feat, s_folder)

           
            avg_psnr += crt_psnr
            avg_ssim += crt_ssim
            logger.info(f'{subfolder_name}--{img_name} - PSNR: {crt_psnr:.6f} dB. SSIM: {crt_ssim:.6f}')
            name_idx += 1
        
        #avg_flops = sum(flops)/len(flops) 
        avg_psnr /= name_idx
        logger.info(f'name_idx:{name_idx}')
        avg_ssim /= name_idx
        avg_psnr_l.append(avg_psnr)
        avg_ssim_l.append(avg_ssim)
        #FLOPs.append(avg_flops)

    for folder_idx, subfolder_name in enumerate(subfolder_names):
        logger.info(f'Folder {subfolder_name} - Average PSNR: {avg_psnr_l[folder_idx]:.6f} dB. Average SSIM: {avg_ssim_l[folder_idx]:.6f}. ')#Average FLOPS: {FLOPs[folder_idx]:.6f}.')
        if measure_inference_time:
            logger.info(f'Folder {subfolder_name} - Inference time: {inference_times[folder_idx]:.2f} ms. ')

    logger.info(f'Average PSNR: {sum(avg_psnr_l) / len(avg_psnr_l):.6f} dB ' f'for {len(subfolder_names)} clips. ')
    logger.info(f'Average SSIM: {sum(avg_ssim_l) / len(avg_ssim_l):.6f}  '
    f'for {len(subfolder_names)} clips. ')

    # --- Log average inference time ---
    if measure_inference_time and inference_times:
        total_frames = sum(num_frames_list)
        total_time_ms = sum(inference_times)
        num_clips = len(inference_times)

        avg_time_per_clip = total_time_ms / num_clips
        # Calculate average time per frame. This is the most accurate way for FPS.
        avg_time_per_frame = total_time_ms / total_frames
        avg_fps = 1000 / avg_time_per_frame

        logger.info(f'==================================================')
        logger.info(f'Performance Summary:')
        logger.info(f'  - Total clips processed: {num_clips}')
        logger.info(f'  - Total frames processed: {total_frames}')
        logger.info(f'  - Total inference time: {total_time_ms / 1000:.2f} s')
        logger.info(f'  - Avg. time per clip: {avg_time_per_clip:.2f} ms')
        logger.info(f'  - Avg. time per frame: {avg_time_per_frame:.2f} ms')
        logger.info(f'  - Average FPS: {avg_fps:.2f}')
        logger.info(f'==================================================')

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


if __name__ == '__main__':

    main()
