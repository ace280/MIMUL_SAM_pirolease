import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# import argparse



# def get_arguments():
    
#     parser = argparse.ArgumentParser()

#     parser.add_argument('-io', '--input_output_directory', type=str, required=True, help='Path to the working directory with inputs and outpus.')
#     parser.add_argument('-ma', '--manufacturer', type=str, required=False, help='The piano roll manufacturer. This argument is for better sorting while testing')
#     parser.add_argument('-t', '--target', type=str, required=True, help='The target that should be segmented. This is for sorting during the test phase.')
#     parser.add_argument('-i', '--input', type=str, required=True, help='The file name (ID) of the image and mask files (without extention) to be used as reference input. Image needs to be JPG, Mask needs to be PNG in their respective folders.')
#     parser.add_argument('-m', '--mode', type=str, required=True, default='box', help='The mode that FastSAM used to create the mask. Needed to find the right folder.')

#     # parser.add_argument('--pred_path', type=str, default='persam')
#     # parser.add_argument('--gt_path', type=str, default='./data/Annotations')

#     # parser.add_argument('--ref_idx', type=str, default='00')
    
#     args = parser.parse_args()
#     return args


def eval_mIoU(input_output_directory, manufacturer, target, mode, id):

    # args = get_arguments()
    # print("Args:", args, "\n"), 

    # class_names = sorted(os.listdir(args.gt_path))
    # class_names = [class_name for class_name in class_names if ".DS" not in class_name]
    # class_names.sort()

    #path preparation
    fastsam_output_path = f'{input_output_directory}/{manufacturer}/Outputs/{target}/FastSAM results/{mode}/Masks'
    persam_output_path = f'{input_output_directory}/{manufacturer}/Outputs/{target}/PerSAM results/{mode}/input_{id}/Masks'
    persam_f_output_path = f'{input_output_directory}/{manufacturer}/Outputs/{target}/PerSAM_F results/{mode}/input_{id}/Masks'

    # persam_mIoU, persam_f_mIoU, persam_mAcc, persam_f_mIoU = 0, 0, 0, 0

    images = os.listdir(fastsam_output_path)

    # compare fastsam masks with persam masks for this input
    # persam_mIoU, persam_mAcc = 0, 0
    count = 0

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for image in images:

        print(f"\nFastSAM input: {id} compared masks: {image}")

        count += 1

        fastsam_mask = f'{fastsam_output_path}/{image}'
        fastsam_mask = cv2.imread(fastsam_mask)
        fastsam_mask = cv2.cvtColor(fastsam_mask, cv2.COLOR_BGR2GRAY) > 0
        plt.imshow(fastsam_mask)
        fastsam_mask = np.uint8(fastsam_mask)
        
        persam_mask = f'{persam_output_path}/{image}'
        persam_mask = cv2.imread(persam_mask)
        persam_mask = cv2.cvtColor(persam_mask, cv2.COLOR_BGR2GRAY) > 0
        plt.imshow(persam_mask)
        persam_mask = np.uint8(persam_mask)

        intersection, union, target = intersectionAndUnion(persam_mask, fastsam_mask)    
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        #print values
        persam_IoU = intersection / union + 1e-10
        persam_Acc = intersection / target + 1e-10

        print("PerSAM IoU: %.2f," %(100 * persam_IoU), "PerSAM Acc: %.2f\n" %(100 * persam_Acc))

    #save values
    persam_mIoU = intersection_meter.sum / (union_meter.sum + 1e-10)
    persam_mAcc = intersection_meter.sum / (target_meter.sum + 1e-10)

    print("PerSAM mIoU: %.2f," %(100 * persam_mIoU), "PerSAM Acc: %.2f\n" %(100 * persam_mAcc))

    # persam_mIoU += persam_IoU
    # persam_mAcc += persam_Acc

    # persam_mIoU = 100 * persam_mIoU / count
    # persam_mAcc = 100 * persam_mAcc / count

    # print("\npersam_mIoU: %.2f" %persam_mIoU)
    # print("persam_mAcc: %.2f\n" %persam_mAcc)

    intersection_meter.reset()
    union_meter.reset()
    target_meter.reset()

    # compare fastsam masks with persam_f masks for this input
    # persam_f_mIoU, persam_f_mAcc = 0, 0
    count = 0

    for image in images:

        print(f"\nFastSAM input: {id} compared masks: {image}")

        count += 1

        fastsam_mask = f'{fastsam_output_path}/{image}'
        fastsam_mask = cv2.imread(fastsam_mask)
        fastsam_mask = cv2.cvtColor(fastsam_mask, cv2.COLOR_BGR2GRAY) > 0
        fastsam_mask = np.uint8(fastsam_mask)

        persam_f_mask = f'{persam_f_output_path}/{image}'
        persam_f_mask = cv2.imread(persam_f_mask)
        persam_f_mask = cv2.cvtColor(persam_f_mask, cv2.COLOR_BGR2GRAY) > 0
        persam_f_mask = np.uint8(persam_f_mask)

        intersection, union, target = intersectionAndUnion(persam_f_mask, fastsam_mask)    
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        #print values
        persam_f_IoU = intersection / (union + 1e-10)
        persam_f_Acc = intersection / (target + 1e-10)

        print("PerSAM_F IoU: %.2f," %(100 * persam_f_IoU), "PerSAM_F Acc: %.2f\n" %(100 * persam_f_Acc))

    #save values
    persam_f_mIoU = intersection_meter.sum / (union_meter.sum + 1e-10)
    persam_f_mAcc = intersection_meter.sum / (target_meter.sum + 1e-10)

    print("PerSAM_F IoU: %.2f," %(100 * persam_f_IoU), "PerSAM_F Acc: %.2f\n" %(100 * persam_f_Acc))
    
    # persam_f_IoU = 100 * persam_f_mIoU / count
    # persam_f_Acc = 100 * persam_f_mAcc / count

    # print("\npersam_f_IoU: %.2f" %persam_f_mIoU)
    # print("persam_f_Acc: %.2f\n" %persam_f_mAcc)

    # persam_mIoU = str(round(persam_mIoU, 2))
    # persam_mAcc  = str(round(persam_mAcc, 2))
    # persam_f_mIoU = str(round(persam_f_mIoU, 2))
    # persam_f_mAcc  = str(round(persam_f_mAcc, 2))
    
    # print(f'persam_mIoU = {persam_mIoU}, persam_mAcc = {persam_mAcc}, persam_f_mIoU = {persam_f_mIoU}, persam_f_mAcc= {persam_f_mAcc}')

    return(persam_mIoU, persam_mAcc, persam_f_mIoU, persam_f_mAcc)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target):
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    
    area_intersection = np.logical_and(output, target).sum()
    area_union = np.logical_or(output, target).sum()
    area_target = target.sum()

    return area_intersection, area_union, area_target

