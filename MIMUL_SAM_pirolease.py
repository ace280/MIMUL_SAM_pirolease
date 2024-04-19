import argparse
from tqdm import tqdm
import csv
import os
import subprocess
import eval_mIoU_MIMUL

def parse_args():

    parser = argparse.ArgumentParser(prog='MIMUL FastSAM', description='Joins FastSAM and PerSAM for segmenting MIMUL piano roll leads.')
    parser.add_argument('-d', '--device', type=str,  required=False, default='cpu', help='The computing device to work on. To work on graphics card use \'CUDA\', default is \'cpu\'')
    parser.add_argument('-io', '--input_output_directory', type=str, required=True, help='Path to the working directory with inputs and outpus.')
    parser.add_argument('-ma', '--manufacturer', type=str, required=True, help='The piano roll manufacturer.')  
    parser.add_argument('-c', '--ckpt', type=str, required=False, default='sam_vit_h_4b8939.pth', help='Needed if another checkpoint shall be used.')
    
    parser.add_argument('-man', '--manual', type=bool, required=False, help='Override flag for manual mode: ID, target and mode need to be provided with arguments. CSV will be ignored.')
    parser.add_argument('-t', '--target', type=str, required=False, help='The object that should be segmented.')
    parser.add_argument('-i', '--input', type=str, required=False, help='The file name of the image (without extention) to be segmented. (Only tested with JPG.)')
    parser.add_argument('-m', '--mode', type=str, required=False, default='box', help='The mode to use for manually marking the location of the label or licence stamp. For box mode type \'box\'. for points mode use \'points\'.')
    parser.add_argument('-b', '--box', type=str, required=False, help='The box of the label or stamp. Requires format like in FastSAM: bbox default shape [0,0,0,0] -> [x1,y1,x2,y2] Example: box = [[1692, 882, 440, 508]]')
    parser.add_argument('-p', '--points', type=str, required=False, help='The points of the label or stamp. Requires format like in FastSAM. points default shapes: single point: [[0,0]] multiple points: [[x1,y1],[x2,y2]]')
    parser.add_argument('-pl', '--point_labels', type=str, required=False, help='The point_label to define which points belong to the foreground and which belong to the background. Requires format like in FastSAM: point_label default [0] [1,0] 0:background, 1:foreground')
  
    args = parser.parse_args()

    return args

def main():

    #add cleaning script to remove dummy files

    if (args.manual):
        print ("Manual mode is not implemented yet.")
    else:
        csv_iput = f'{args.input_output_directory}/{args.manufacturer}/Input/CSV'
        csv_output = f'{args.input_output_directory}/{args.manufacturer}/Outputs/CSV'
        for csv_file in tqdm(os.listdir(csv_iput)):
            if '.csv' in csv_file or '.CSV' in csv_iput:
                csv_segmentation(csv_iput, csv_output, csv_file)
            else:
                print (f'The file {csv_file} is not a CSV-file. Please only put CSV-files into the CSV input folder.')
                        
def csv_segmentation(csv_iput, csv_output, csv_file):

    os.makedirs(csv_output, exist_ok=True)

    target = csv_file.split('.')[0]
    print (f"Found CSV for target {target}.")
    print (f"Input and Outputs directory set to {args.input_output_directory}")

    csv_input_path = os.path.join(csv_iput, csv_file)
    csv_output_path = os.path.join(csv_output, csv_file)

    # add cleaner for header and clean dummy entries in first line

    print (f"\nStep 1: Segmenting target {target} with FastSAM")
    with open(csv_input_path, newline='', encoding='utf-8-sig') as instructions_csv:
        instructions_reader = csv.DictReader(instructions_csv, dialect='excel', delimiter=';')
        with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as resumption_csv:
            resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
            resumption_writer.writeheader()
        for image_row in instructions_reader:
            if image_row['done'] == '':
                fastSAM(target, image_row)
                image_row['done'] = 1
            else:
                print (f"Step 1 skipped for image {image_row['image']}.")
            with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as resumption_csv:
                resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
                resumption_writer.writerow(image_row)
    os.replace(csv_output_path, csv_input_path)

    print ("\nStep 2: Using FastSAM masks for PerSAM extraction")
    with open(csv_input_path, newline='', encoding='utf-8-sig') as instructions_csv:
        instructions_reader = csv.DictReader(instructions_csv, dialect='excel', delimiter=';')
        with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as resumption_csv:
            resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
            resumption_writer.writeheader()
        for image_row in instructions_reader:
            if image_row['done'] == '' or int(image_row['done']) <= 1:
                perSAM(target, image_row)
                image_row['done'] = 2
            else:
                print (f"Step 2 skipped for image {image_row['image']}.")
            with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as resumption_csv:
                resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
                resumption_writer.writerow(image_row)
    os.replace(csv_output_path, csv_input_path)

    print ("\nStep 3: Using FastSAM masks for PerSAM_f extraction (with some training).")
    with open(csv_input_path, newline='', encoding='utf-8-sig') as instructions_csv:
        instructions_reader = csv.DictReader(instructions_csv, dialect='excel', delimiter=';')
        with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as resumption_csv:
            resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
            resumption_writer.writeheader()
        for image_row in instructions_reader:
            if image_row['done'] == '' or int(image_row['done']) <= 2:
                perSAM_F(target, image_row)
                image_row['done'] = 3
            else:
                print (f"Step 3 skipped for image {image_row['image']}.")
            with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as resumption_csv:
                resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
                resumption_writer.writerow(image_row)
    os.replace(csv_output_path, csv_input_path)

    print ("\nStep 4: Evaluating IoU and Accuracy between FastSAM and PerSAM masks.")
    persam_mIoU, persam_mAcc, persam_f_mIoU, persam_f_mAcc  = 0, 0, 0, 0
    count = 0
    with open(csv_input_path, newline='', encoding='utf-8-sig') as instructions_csv:
        instructions_reader = csv.DictReader(instructions_csv, dialect='excel', delimiter=';')
        with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as resumption_csv:
            resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
            resumption_writer.writeheader()
        for image_row in instructions_reader:
            if int(image_row['done']) <= 3:
                persam_IoU, persam_Acc, persam_f_IoU, persam_f_Acc = eval_mIoU(target, image_row)
                image_row['done'] = 4
                image_row['persam_IoU'] = str(round(persam_IoU, 2))
                image_row['persam_Acc'] = str(round(persam_Acc, 2))
                image_row['persam_f_IoU'] = str(round(persam_f_IoU, 2))
                image_row['persam_f_Acc'] = str(round(persam_f_Acc, 2))
                persam_mIoU += persam_IoU
                persam_mAcc += persam_Acc
                persam_f_mIoU += persam_f_IoU
                persam_f_mAcc += persam_f_Acc
                count +=1
            else:
                print (f"Step 4 skipped for image {image_row['image']}.")
            with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as resumption_csv:
                resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
                resumption_writer.writerow(image_row)
    os.replace(csv_output_path, csv_input_path)

    if count > 0:

        persam_mIoU = str(round(persam_mIoU / count, 2))
        persam_mAcc = str(round(persam_mAcc / count, 2))

        persam_f_mIoU = str(round(persam_f_mIoU / count, 2))
        persam_f_mAcc = str(round(persam_f_mAcc / count, 2))

        print(f"\nOverall PerSAM_mIoU for target {target}: {persam_mIoU}")
        print(f"Overall PerSAM_mAcc for target {target}: {persam_mAcc}")

        print(f"\nOverall PerSAM_F_mIoU for target {target}: {persam_f_mIoU}")
        print(f"Overall PerSAM_F_mAcc for target {target}: {persam_f_mAcc}")

        overall_eval_path = f'{args.input_output_directory}/{args.manufacturer}/Outputs/{target}/overall_eval.csv'
        with open(overall_eval_path, 'w', newline='', encoding='utf-8-sig') as eval_csv:
            fieldnames = ['target', 'persam_mIoU', 'persam_mAcc', 'persam_f_mIoU', 'persam_f_mAcc']
            eval_writer = csv.DictWriter(eval_csv, fieldnames=fieldnames, dialect='excel', delimiter=';')
            eval_writer.writeheader()
            eval_writer.writerow({'target': target, 'persam_mIoU': persam_mIoU, 'persam_mAcc': persam_mAcc, 'persam_f_mIoU': persam_f_mIoU, 'persam_f_mAcc': persam_f_mAcc})

def fastSAM(target, image_row):

    id = image_row['image'].strip('.jpg')
    mode = image_row['mode']
    mode_details = ""
    points = ""
    point_labels = ""

    print (f"Segmenting {target} from {id} of manufacturer {args.manufacturer} using {mode} mode.")

    if (mode == 'box'):
        box = image_row['box']
        mode_details = f"-b \"{box}\""
        print (f"Box input is {box}. Added mode details: \"{mode_details}\"")
    elif (mode == 'points'):
        points = image_row['points']
        point_labels = image_row['point_labels']
        mode_details = f"-p `\"{points}\" -pl \"{point_labels}\""
        print (f"Points input is \"{points}\" point labels are \"{point_labels}\". Added mode details: \"{mode_details}\"")
    else:
        print (f"Mode {mode} not supported. Please check spelling and choose either \"box\" or \"points\".")

    print (f"\nCalling python \".\FastSAM_MIMUL.py\" -d {args.device} -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode} {mode_details}")
    subprocess.run(f"python \".\FastSAM_MIMUL.py\" -d {args.device} -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode} {mode_details}")

def perSAM(target, image_row):
    id = image_row['image'].strip('.jpg')
    mode = image_row['mode']

    print (f"\nTesting with input {id} and FastSAM mask generated in {mode} mode for target {target}.")

    print (f"Calling python \".\PerSAM_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    subprocess.run(f"python \".\perSAM_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")

def perSAM_F(target, image_row):
    id = image_row['image'].strip('.jpg')
    mode = image_row['mode']

    print (f"\nTesting with input {id} and FastSAM mask generated in {mode} mode for target {target}.")

    print (f"\nCalling python \".\PerSAM_F_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    subprocess.run(f"python \".\PerSAM_F_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")

def eval_mIoU(target, image_row):
    id = image_row['image'].strip('.jpg')
    mode = image_row['mode']

    print (f"Evaluating PerSAM and PerSAM_F results for ID {id} with FastSAM mask generated in {mode} mode for target {target}.")

    # print (f"\nCalling python \".\eval_mIoU_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    # IoU = subprocess.check_output(f"python \".\eval_mIoU_MIMUL.py\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    
    print (f"\npersam_IoU, persam_Acc, persam_f_IoU, persam_f_Acc = eval_mIoU_MIMUL.eval_mIoU({args.input_output_directory}, {args.manufacturer}, {target}, {mode}, {id})")
    persam_IoU, persam_Acc, persam_f_IoU, persam_f_Acc = eval_mIoU_MIMUL.eval_mIoU(args.input_output_directory, args.manufacturer, target, mode, id)

    return(persam_IoU, persam_Acc, persam_f_IoU, persam_f_Acc)

if __name__ == "__main__":
    args = parse_args()
    main()