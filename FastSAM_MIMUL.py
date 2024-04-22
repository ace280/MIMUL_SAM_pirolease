import os
from fastsam import FastSAM, FastSAMPrompt
import matplotlib.pyplot as plt
import argparse
from utils.tools import convert_box_xywh_to_xyxy
import ast
import cv2
from PIL import Image
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(prog='MIMUL FastSAM', description='Implements FastSAM for use on MIMUL Piano Roll heads.')
    parser.add_argument('-d', '--device', type=str,  required=False, default='cpu', help='The computing device to work on. To work on graphics card use \'CUDA\', default is \'cpu\'')
    parser.add_argument('-io', '--input_output_directory', type=str, required=True, help='Path to the working directory with inputs and outpus.')
    parser.add_argument('-ma', '--manufacturer', type=str, required=True, help='The piano roll manufacturer.')
    parser.add_argument('-t', '--target', type=str, required=True, help='The object that should be segmented.')
    parser.add_argument('-i', '--input', type=str, required=True, help='The file name of the picture (without extention) to be segmented. (Only tested with JPG.)')
    parser.add_argument('-m', '--mode', type=str, required=True, default='box', help='The mode to use for manually marking the location of the label or licence stamp. For box mode type \'box\'. for points mode use \'points\'.')
    parser.add_argument('-b', '--box', type=str, required=False, help='The box of the label or stamp. Requires format like in FastSAM: bbox default shape [0,0,0,0] -> [x1,y1,x2,y2] Example: box = [[1692, 882, 440, 508]]')
    parser.add_argument('-p', '--points', type=str, required=False, help='The points of the label or stamp. Requires format like in FastSAM. points default shapes: single point: [[0,0]] multiple points: [[x1,y1],[x2,y2]]')
    parser.add_argument('-pl', '--point_labels', type=str, required=False, help='The point_label to define which points belong to the foreground and which belong to the background. Requires format like in FastSAM: point_label default [0] [1,0] 0:background, 1:foreground')
   
    args = parser.parse_args()
    print(f'args={args}')

    print(f'Image to be processed: {args.input}')
    if(args.mode == 'box' and args.box != None):
        print(f'Box mode chosen; box={args.box}')
        args.box = convert_box_xywh_to_xyxy(ast.literal_eval(args.box))
    elif(args.mode == 'box' and args.box == None):
        parser.error(f'Box mode requires the --box argument')
    elif(args.mode == 'points' and args.points !=None and args.point_labels != None):
        args.points = ast.literal_eval(args.points)
        args.point_labels = ast.literal_eval(args.point_labels)
        print(f'points mode chosen; points={args.points}, point_labels={args.point_labels}')
    elif(args.mode == 'points' and args.points == None):
        parser.error('Points mode requires the --points argument')
    elif(args.mode == 'points' and args.point_labels == None):
        parser.error('Points mode requires the --point_labels argument')
    
    return args

def main():

    model = FastSAM('./weights/FastSAM-x.pt')
    image_path = f'{args.input_output_directory}/{args.manufacturer}/Input/{args.input}.jpg'
    output_path = f'{args.input_output_directory}/{args.manufacturer}/Outputs/{args.target}/FastSAM results/{args.mode}/'
    # DEVICE = 'CUDA'
    everything_results = model(image_path, device=args.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(image_path, everything_results, device=args.device)

    if args.mode=='box': 
        # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=list(args.box))   

    if args.mode=='points':
        # point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        ann = prompt_process.point_prompt(points=list(args.points), pointlabel=list(args.point_labels))

    save_mask(ann,output_path)

    #save reference image
    os.makedirs(f'{output_path}/Images/', exist_ok=True)
    print(f"Promting FastSAM for {args.input} using {args.mode}-promt")
    prompt_process.plot(annotations=ann, output_path=f'{output_path}/Images/{args.input}.jpg',withContours=True)


def save_mask(ann, output_path):

    for i, mask in enumerate(ann):
        if type(mask) == dict:
            mask = mask['segmentation']

        os.makedirs(f'{output_path}/Masks/', exist_ok=True)
        print(f"saving annotations mask {i} to folder {output_path}/Masks/")

        plt.imsave(f'{output_path}/Masks/{args.input}.png', mask)
        plt.close

        im = Image.open(f'{output_path}/Masks/{args.input}.png')
        data = np.array(im)

        r1, g1, b1 = 68, 1, 84 # Original value
        r2, g2, b2 = 0, 0, 0 # Value that we want to replace it with

        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data[:,:,:3][mask] = [r2, g2, b2]

        r1, g1, b1 = 253, 231, 36 # Original value
        r2, g2, b2 = 128, 0, 0 # Value that we want to replace it with

        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data[:,:,:3][mask] = [r2, g2, b2]

        im = Image.fromarray(data)
        im.save(f'{output_path}/Masks/{args.input}.png')


if __name__ == "__main__":
    args = parse_args()
    main()