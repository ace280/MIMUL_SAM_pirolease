import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def eval_mIoU(input_output_directory, manufacturer, target, mode, id):

    #path preparation
    fastsam_output_path = f'{input_output_directory}/{manufacturer}/Outputs/{target}/FastSAM results/{mode}/Masks'
    persam_output_path = f'{input_output_directory}/{manufacturer}/Outputs/{target}/PerSAM results/{mode}/input_{id}/Masks'
    persam_f_output_path = f'{input_output_directory}/{manufacturer}/Outputs/{target}/PerSAM_F results/{mode}/input_{id}/Masks'

    images = os.listdir(fastsam_output_path)

    # compare fastsam masks with persam masks for this input
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
        fastsam_mask = np.uint8(fastsam_mask)
        
        persam_mask = f'{persam_output_path}/{image}'
        persam_mask = cv2.imread(persam_mask)
        persam_mask = cv2.cvtColor(persam_mask, cv2.COLOR_BGR2GRAY) > 0
        persam_mask = np.uint8(persam_mask)

        intersection, union, target = intersectionAndUnion(persam_mask, fastsam_mask)    
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        #print values
        persam_IoU = intersection / (union + 1e-10)
        persam_Acc = intersection / (target + 1e-10)

        print("PerSAM IoU: %.2f," %(100 * persam_IoU), "PerSAM Acc: %.2f\n" %(100 * persam_Acc))

    #save values
    persam_mIoU = intersection_meter.sum / (union_meter.sum + 1e-10)
    persam_mAcc = intersection_meter.sum / (target_meter.sum + 1e-10)

    print("PerSAM mIoU: %.2f," %(100 * persam_mIoU), "PerSAM Acc: %.2f\n" %(100 * persam_mAcc))

    intersection_meter.reset()
    union_meter.reset()
    target_meter.reset()

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
    plt.imshow(target)
    plt.imshow(output)

    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)    

    intersection = np.logical_and(target, output)
    union = np.logical_or(output, target)
    
    area_intersection = intersection.sum()
    area_union = union.sum()
    area_target = target.sum()

    return area_intersection, area_union, area_target

