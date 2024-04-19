import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import csv

def eval_mIoU(input_output_directory, manufacturer, target, mode, id):

    #path preparation
    fastsam_output_path = f'{input_output_directory}/{manufacturer}/Outputs/{target}/FastSAM results/{mode}'
    persam_output_path = f'{input_output_directory}/{manufacturer}/Outputs/{target}/PerSAM results/{mode}/input_{id}'
    persam_f_output_path = f'{input_output_directory}/{manufacturer}/Outputs/{target}/PerSAM_F results/{mode}/input_{id}'

    images = os.listdir(f'{fastsam_output_path}/Masks')

    # compare fastsam masks with persam masks for this input

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    count = 0
    persam_mIoU = 0
    persam_mAcc = 0

    for image in images:

        print(f"\nFastSAM input: {id} compared masks: {image}")

        count += 1

        fastsam_mask = f'{fastsam_output_path}/Masks/{image}'
        fastsam_mask = cv2.imread(fastsam_mask)
        fastsam_mask = cv2.cvtColor(fastsam_mask, cv2.COLOR_BGR2GRAY) > 0
        fastsam_mask = np.uint8(fastsam_mask)
        
        persam_mask = f'{persam_output_path}/Masks/{image}'
        persam_mask = cv2.imread(persam_mask)
        persam_mask = cv2.cvtColor(persam_mask, cv2.COLOR_BGR2GRAY) > 0
        persam_mask = np.uint8(persam_mask)

        intersection, union, target = intersectionAndUnion(persam_mask, fastsam_mask)    
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        #print values
        persam_IoU = intersection / (union + 1e-10)
        persam_Acc = intersection / (target + 1e-10)

        print("PerSAM IoU: %.2f," %(100 * persam_IoU), "PerSAM Acc: %.2f\n" %(100 * persam_Acc))

        csv_output_path = f"{persam_output_path}/Evaluation.csv"

        if count == 1:
            with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as eval_csv:
                fieldnames = ['image', 'persam_IoU', 'persam_Acc']
                eval_writer = csv.DictWriter(eval_csv, fieldnames=fieldnames, dialect='excel', delimiter=';')
                eval_writer.writeheader()

        with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as eval_csv:
            fieldnames = ['image', 'persam_IoU', 'persam_Acc']
            eval_writer = csv.DictWriter(eval_csv, fieldnames=fieldnames, dialect='excel', delimiter=';')
            eval_writer.writerow({'image': image, 'persam_IoU': persam_IoU, 'persam_Acc': persam_Acc})

        #calculate m-Values by hand
        persam_mIoU += persam_IoU
        persam_mAcc += persam_Acc

    #save values
    persam_mIoU_bh = 100 * persam_mIoU / count
    persam_mAcc_bh = 100 * persam_mAcc / count
    
    # print("\nPerSAM mIoU by hand = %.2f" %(100 * persam_mIoU / count), "PerSAM mAcc by hand = %.2f\n" %(100 * persam_mAcc / count))
    print("\nPerSAM mIoU by hand = %.2f" %persam_mIoU_bh, "PerSAM mAcc by hand = %.2f\n" %(persam_mAcc_bh))


    persam_mIoU_wm = intersection_meter.sum / (union_meter.sum + 1e-10)
    persam_mAcc_wm = intersection_meter.sum / (target_meter.sum + 1e-10)

    print("PerSAM mIoU with meters: %.2f," %(100 * persam_mIoU_wm), "PerSAM mAcc with meters: %.2f\n" %(100 * persam_mAcc_wm))

    intersection_meter.reset()
    union_meter.reset()
    target_meter.reset()

    count = 0
    persam_f_mIoU = 0
    persam_f_mAcc = 0

    for image in images:

        # intersection summieren und dann per hand noch mal teilen und mit averagemeter-Werten vergleichen

        print(f"\nFastSAM input: {id} compared masks: {image}")

        count += 1

        fastsam_mask = f'{fastsam_output_path}/Masks/{image}'
        fastsam_mask = cv2.imread(fastsam_mask)
        fastsam_mask = cv2.cvtColor(fastsam_mask, cv2.COLOR_BGR2GRAY) > 0
        fastsam_mask = np.uint8(fastsam_mask)

        persam_f_mask = f'{persam_f_output_path}/Masks/{image}'
        persam_f_mask = cv2.imread(persam_f_mask)
        persam_f_mask = cv2.cvtColor(persam_f_mask, cv2.COLOR_BGR2GRAY) > 0
        persam_f_mask = np.uint8(persam_f_mask)

        intersection, union, target = intersectionAndUnion(persam_f_mask, fastsam_mask)    
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        #print values
        persam_f_IoU = intersection / (union + 1e-10)
        persam_f_Acc = intersection / (target + 1e-10)

        print("PerSAM_F IoU: %.2f," %(100 * persam_f_IoU), "PerSAM_F Acc: %.2f\n" %(100 * persam_f_Acc))

        csv_output_path = f"{persam_f_output_path}/Evaluation.csv"

        if count == 1:
            with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as eval_csv:
                fieldnames = ['image', 'persam_f_IoU', 'persam_f_Acc']
                eval_writer = csv.DictWriter(eval_csv, fieldnames=fieldnames, dialect='excel', delimiter=';')
                eval_writer.writeheader()

        with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as eval_csv:
            fieldnames = ['image', 'persam_f_IoU', 'persam_f_Acc']
            eval_writer = csv.DictWriter(eval_csv, fieldnames=fieldnames, dialect='excel', delimiter=';')
            eval_writer.writerow({'image': image, 'persam_f_IoU': persam_f_IoU, 'persam_f_Acc': persam_f_Acc})

        #calculate m-Values by hand
        persam_f_mIoU += persam_f_IoU
        persam_f_mAcc += persam_f_Acc

    persam_f_mIoU_bh = 100 * persam_f_mIoU / count
    persam_f_mAcc_bh = 100 * persam_f_mAcc / count

    print("\nPerSAM_F mIoU by hand = %.2f" %(persam_f_mIoU_bh), "PerSAM_F mAcc by hand = %.2f\n" %(persam_f_mAcc_bh))

    #save values
    persam_f_mIoU_wm = intersection_meter.sum / (union_meter.sum + 1e-10)
    persam_f_mAcc_wm = intersection_meter.sum / (target_meter.sum + 1e-10)

    print("PerSAM_F mIoU with meters: %.2f," %(100 * persam_f_mIoU_wm), "PerSAM_F mAcc with meters: %.2f\n" %(100 * persam_f_mAcc_wm))
    
    return(persam_mIoU_bh, persam_mAcc_bh, persam_f_mIoU_bh, persam_f_mAcc_bh)


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

    intersection = np.logical_and(output, target)
    union = np.logical_or(output, target)
    
    area_intersection = intersection.sum()
    area_union = union.sum()
    area_target = target.sum()

    return area_intersection, area_union, area_target

