import argparse
from tqdm import tqdm
import csv
import os
import subprocess

def parse_args():

    parser = argparse.ArgumentParser(prog='MIMUL FastSAM', description='Joins FastSAM and PerSAM for segmenting MIMUL piano roll leads.')
    parser.add_argument('-d', '--device', type=str,  required=False, default='cpu', help='The computing device to work on. To work on graphics card use \'CUDA\', default is \'cpu\'')
    parser.add_argument('-io', '--input_output_directory', type=str, required=True, help='Path to the working directory with inputs and outpus.')
    parser.add_argument('-ma', '--manufacturer', type=str, required=True, help='The piano roll manufacturer.')  
    parser.add_argument('-c', '--ckpt', type=str, required=False, default='sam_vit_h_4b8939.pth', help='Needed if another checkpoint shall be used.')
    
    parser.add_argument('-man', '--manual', type=bool, required=False, help='Override flag for manual mode: ID, target and mode need to be provided with arguments. CSV will be ignored.')
    parser.add_argument('-t', '--target', type=str, required=False, help='The object that should be segmented.')
    parser.add_argument('-i', '--input', type=str, required=False, help='The file name of the picture (without extention) to be segmented. (Only tested with JPG.)')
    parser.add_argument('-m', '--mode', type=str, required=False, default='box', help='The mode to use for manually marking the location of the label or licence stamp. For box mode type \'box\'. for points mode use \'points\'.')
    parser.add_argument('-b', '--box', type=str, required=False, help='The box of the label or stamp. Requires format like in FastSAM: bbox default shape [0,0,0,0] -> [x1,y1,x2,y2] Example: box = [[1692, 882, 440, 508]]')
    parser.add_argument('-p', '--points', type=str, required=False, help='The points of the label or stamp. Requires format like in FastSAM. points default shapes: single point: [[0,0]] multiple points: [[x1,y1],[x2,y2]]')
    parser.add_argument('-pl', '--point_labels', type=str, required=False, help='The point_label to define which points belong to the foreground and which belong to the background. Requires format like in FastSAM: point_label default [0] [1,0] 0:background, 1:foreground')
  
    args = parser.parse_args()

    return args

def main():

    if (args.manual):
        print ("Manual mode is not implemented yet.")
    else:
        csv_path = f'{args.input_output_directory}/{args.manufacturer}/Input/CSV'
        for csv_file in tqdm(os.listdir(csv_path)):
            if '.csv' in csv_file or '.CSV' in csv_file:
                csv_segmentation(csv_path, csv_file)
            else:
                print (f'The file {csv_file} is not a CSV-file. Please only put CSV-files into the CSV input folder.')
                        
def csv_segmentation(csv_path, csv_file):

    target = csv_file.split('.')[0]
    print (f"Found CSV {target} and set it as target.")
    print (f"Input and Outputs directory set to {args.input_output_directory}")

    csv_file_path = os.path.join(csv_path, csv_file)

    print (f"\nStep 1: Segmenting target {target} with FastSAM")
    with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel', delimiter=';')

        for image in reader:
            print (f"Step 1 skipped for image {image}.")
            # fastSAM(target, image)
        

    print ("\nStep 2: Using FastSAM masks for PerSAM extraction")
    with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel', delimiter=';')
            
        for image in reader:
            print (f"Step 2 skipped for image {image}.")
            # perSAM(target, image)

    print ("\nStep 3: Using FastSAM masks for PerSAM_f extraction (with some training).")
    with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel', delimiter=';')

        for image in reader:
            # print (f"Step 3 skipped for image {image}.")
            perSAM_f(target, image)

def fastSAM(target, image):

    id = image['ID']
    mode = image['mode']
    mode_details = ""
    points = ""
    point_labels = ""

    print (f"Segmenting {target} from {id} of manufacturer {args.manufacturer} using {mode} mode.")

    if (mode == 'box'):
        box = image['box']
        mode_details = f"-b \"{box}\""
        print (f"Box input is {box}. Added mode details: \"{mode_details}\"")
    elif (mode == 'points'):
        points = image['points']
        point_labels = image['point_labels']
        mode_details = f"-p `\"{points}\" -pl \"{point_labels}\""
        print (f"Points input is \"{points}\" point labels are \"{point_labels}\". Added mode details: \"{mode_details}\"")
    else:
        print (f"Mode {mode} not supported. Please check spelling and choose either \"box\" or \"points\".")

    print (f"\nCalling python \".\FastSAM_MIMUL.py\" -d {args.device} -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode} {mode_details}")
    subprocess.run(f"python \".\FastSAM_MIMUL.py\" -d {args.device} -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode} {mode_details}")

def perSAM(target, image):
    id = image['ID']
    mode = image['mode']

    print (f"\nTesting with input {id} and FastSAM mask generated in {mode} mode for target {target}.")

    print (f"Calling python \".\PerSAM_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    subprocess.run(f"python \".\perSAM_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")

def perSAM_f(target, image):
    id = image['ID']
    mode = image['mode']

    print (f"Testing with input {id} and FastSAM mask generated in {mode} mode for target {target}.")

    print (f"\nCalling python \".\PerSAM_f_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    subprocess.run(f"python \".\PerSAM_f_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")



if __name__ == "__main__":
    args = parse_args()
    main()