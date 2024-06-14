import argparse
import csv
import subprocess
import eval_mIoU_MIMUL
import logging
from pathlib import Path
import sys

def parse_args():
    global args
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

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as aae:
         print(f"Error parsing arguments: {aae}")

def setup_logging(new_path=None, target=None):
    if new_path == None:
        try:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                                filename=f'{args.input_output_directory}/pirolease_startup.log',
                                filemode='a')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)
        except logging.exception as le:
            print(f"Exception while trying to start logging. Error: {le}")
        except FileNotFoundError as fnfe:
            print(f"Error while trying to create or log file could not be found. Error: {fnfe}")
        except PermissionError as pe:
            print(f"Error with permissions of the logging file. Error: {pe}")
        except Exception as e:
            print(f"Error while trying to start logging. Error: {e}")
    else:
        new_path = Path(new_path)
        try: 
            new_path.mkdir(parents=True, exist_ok=True)
            new_path = new_path / f"{target}.log"
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logging.getLogger().removeHandler(handler)
            logging.basicConfig(force=True, filename=new_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', filemode='a')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)
        except OSError as ose:
            logging.error (f"Error creating directories {new_path}. Error: {ose}") 
        except Exception as e:
            logging.error (f"Error while trying to change logging file. Error: {e}")

def main():
    try:
        if (args.manual):
            logging.info("Manual mode is not implemented yet.")
        else:
            csv_input_folder = Path(f'{args.input_output_directory}/{args.manufacturer}/Input/CSV')
            csv_output_folder = Path(f'{args.input_output_directory}/{args.manufacturer}/Outputs/CSV')
            for csv_file in csv_input_folder.iterdir():
                if csv_file.suffix.lower() == '.csv':
                    target = csv_file.stem
                    csv_output = Path(csv_output_folder, f'{target}.csv')
                    if not csv_output_folder.exists():
                        try:
                            csv_output_folder.mkdir(parents=True, exist_ok=True)
                        except OSError as ose:
                            logging.error (f"Error creating directories {csv_output_folder}. Error: {ose}") 
                    csv_segmentation(csv_file, csv_output, target)
                else:
                    logging.warning (f'The file {csv_file} is not a CSV-file. Please only put CSV-files into the CSV input folder.')
    except Exception as e:
        logging.error (f"An unexpected error occurred in MIMUL_SAM_pirolese. Error: {e}")      

def csv_segmentation(csv_input_path, csv_output_path, target):

    logging.info (f"Found CSV for target {target}.")
    logging.info (f"Input and Outputs directory set to {args.input_output_directory}")
    logging.info (f"Logging will now switch logfile output to the target folder.")

    setup_logging(f"{args.input_output_directory}/{args.manufacturer}/Outputs/{target}", target)

    logging.info (f"\nStep 1: Segmenting target {target} with FastSAM")
    try:
        with open(csv_input_path, newline='', encoding='utf-8-sig') as instructions_csv:
            instructions_reader = csv.DictReader(instructions_csv, dialect='excel', delimiter=';')
            with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as resumption_csv:
                resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
                resumption_writer.writeheader()
            for image_row in instructions_reader:
                if image_row['done'] == '' or image_row['done'] == None:
                    result = fastSAM(target, image_row)
                    if result.returncode == 0:
                        image_row['done'] = 1
                else:
                    logging.info (f"Step 1 already done for image {image_row['image']}, skipped.")
                with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as resumption_csv:
                    resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
                    resumption_writer.writerow(image_row)
        try:
            csv_output_path.replace(csv_input_path)
        except PermissionError as pe:
            logging.error(f"Error with permission when trying to rename the file: {pe}")
    except FileNotFoundError as fnfe:
        logging.error(f"CSV file {csv_input_path} could not be found. Error: {fnfe}")
    except IOError as ioe:
        logging.error(f"I/O error while trying to read CSV file {csv_input_path}. Error: {ioe}")
    except Exception as e: 
        logging.error (f"Unexpected error while trying to read CSV file: {e}")


    logging.info ("\nStep 2: Using FastSAM masks for PerSAM extraction")
    try:
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
                    logging.info (f"Step 2 already done for image {image_row['image']}, skipped.")
                with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as resumption_csv:
                    resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
                    resumption_writer.writerow(image_row)
        try:
            csv_output_path.replace(csv_input_path)
        except PermissionError as pe:
            logging.error(f"Error with permission when trying to rename the file: {pe}")
    except FileNotFoundError as fnfe:
        logging.error(f"CSV file {csv_input_path} could not be found. Error: {fnfe}")
    except IOError as ioe:
        logging.error(f"I/O error while trying to read CSV file {csv_input_path}. Error: {ioe}")
    except Exception as e: 
        logging.error (f"Unexpected error while trying to read CSV file: {e}")

    logging.info ("\nStep 3: Using FastSAM masks for PerSAM_f extraction (with some training).")
    try:
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
                    logging.info (f"Step 3 already done for image {image_row['image']}, skipped.")
                with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as resumption_csv:
                    resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
                    resumption_writer.writerow(image_row)
        try:
            csv_output_path.replace(csv_input_path)
        except PermissionError as pe:
            logging.error(f"Error with permission when trying to rename the file: {pe}")
    except FileNotFoundError as fnfe:
        logging.error(f"CSV file {csv_input_path} could not be found. Error: {fnfe}")
    except IOError as ioe:
        logging.error(f"I/O error while trying to read CSV file {csv_input_path}. Error: {ioe}")
    except Exception as e: 
        logging.error (f"Unexpected error while trying to read CSV file: {e}")

    logging.info ("\nStep 4: Evaluating IoU and Accuracy between FastSAM and PerSAM masks.")
    try:
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
                    logging.info (f"Step 4 already done for image {image_row['image']}, skipped.")
                with open(csv_output_path, 'a', newline='', encoding='utf-8-sig') as resumption_csv:
                    resumption_writer = csv.DictWriter(resumption_csv, fieldnames=instructions_reader.fieldnames, dialect='excel', delimiter=';')
                    resumption_writer.writerow(image_row)
        try:
            csv_output_path.replace(csv_input_path)
        except PermissionError as pe:
            logging.error(f"Error with permission when trying to rename the file: {pe}")
    except FileNotFoundError as fnfe:
        logging.error(f"CSV file {csv_input_path} could not be found. Error: {fnfe}")
    except IOError as ioe:
        logging.error(f"I/O error while trying to read CSV file {csv_input_path}. Error: {ioe}")
    except Exception as e: 
        logging.error (f"Unexpected error while trying to read CSV file: {e}")

    if count > 0:

        persam_mIoU = str(round(persam_mIoU / count, 2))
        persam_mAcc = str(round(persam_mAcc / count, 2))

        persam_f_mIoU = str(round(persam_f_mIoU / count, 2))
        persam_f_mAcc = str(round(persam_f_mAcc / count, 2))

        logging.info(f"\nOverall PerSAM_mIoU for target {target}: {persam_mIoU}")
        logging.info(f"Overall PerSAM_mAcc for target {target}: {persam_mAcc}")

        logging.info(f"\nOverall PerSAM_F_mIoU for target {target}: {persam_f_mIoU}")
        logging.info(f"Overall PerSAM_F_mAcc for target {target}: {persam_f_mAcc}")

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

    logging.info (f"Segmenting {target} from {id} of manufacturer {args.manufacturer} using {mode} mode.")

    if (mode == 'box'):
        box = image_row['box']
        mode_details = f"-b \"{box}\""
        logging.info (f"Box input is {box}. Added mode details: \"{mode_details}\"")
    elif (mode == 'points'):
        points = image_row['points']
        point_labels = image_row['point_labels']
        mode_details = f"-p \"{points}\" -pl \"{point_labels}\""
        logging.info (f"Points input is \"{points}\" point labels are \"{point_labels}\". Added mode details: \"{mode_details}\"")
    else:
        logging.warning (f"Mode {mode} not supported. Please check spelling and choose either \"box\" or \"points\".")

    logging.info (f"Calling python \".\FastSAM_MIMUL.py\" -d {args.device} -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode} {mode_details}")
    try:
        return subprocess.run(f"python \".\FastSAM_MIMUL.py\" -d {args.device} -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode} {mode_details}")
    except subprocess.CalledProcessError as cpe:
        logging.error(f"Error while excecuting FastSAM subprocess with input {id} and FastSAM mask generated in {mode} mode for target {target}. Error: {cpe}")
        return -1
    except Exception as e: 
        logging.error (f"Unexpected error while trying to execute FastSAM subprocess with input {id} and FastSAM mask generated in {mode} mode for target {target}. Error: {e}")
        return -1

def perSAM(target, image_row):
    id = image_row['image'].strip('.jpg')
    mode = image_row['mode']

    logging.info (f"\nTesting with input {id} and FastSAM mask generated in {mode} mode for target {target}.")

    logging.info (f"Calling python \".\PerSAM_MIMUL.py\" -d \"{args.device}\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    try:
        subprocess.run(f"python \".\perSAM_MIMUL.py\" -d \"{args.device}\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    except subprocess.CalledProcessError as cpe:
        logging.error(f"Error while excecuting PerSAM subprocess with input {id} and FastSAM mask generated in {mode} mode for target {target}. Error: {cpe}")
    except Exception as e: 
        logging.error (f"Unexpected error while trying to execute PerSAM subprocess with input {id} and FastSAM mask generated in {mode} mode for target {target}. Error: {e}")


def perSAM_F(target, image_row):
    id = image_row['image'].strip('.jpg')
    mode = image_row['mode']

    logging.info (f"\nTesting with input {id} and FastSAM mask generated in {mode} mode for target {target}.")

    logging.info (f"\nCalling python \".\PerSAM_F_MIMUL.py\" -d \"{args.device}\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    try:
        subprocess.run(f"python \".\PerSAM_F_MIMUL.py\" -d \"{args.device}\" -io \"{args.input_output_directory}\" -ma \"{args.manufacturer}\" -t {target} -i {id} -m {mode}")
    except subprocess.CalledProcessError as cpe:
        logging.error(f"Error while excecuting PerSAM subprocess with input {id} and FastSAM mask generated in {mode} mode for target {target}. Error: {cpe}")
    except Exception as e: 
        logging.error (f"Unexpected error while trying to execute PerSAM subprocess with input {id} and FastSAM mask generated in {mode} mode for target {target}. Error: {e}")


def eval_mIoU(target, image_row):
    id = image_row['image'].strip('.jpg')
    mode = image_row['mode']

    logging.info (f"Evaluating PerSAM and PerSAM_F results for ID {id} with FastSAM mask generated in {mode} mode for target {target}.")

    logging.info (f"\npersam_IoU, persam_Acc, persam_f_IoU, persam_f_Acc = eval_mIoU_MIMUL.eval_mIoU({args.input_output_directory}, {args.manufacturer}, {target}, {mode}, {id})")
    try:
        persam_IoU, persam_Acc, persam_f_IoU, persam_f_Acc = eval_mIoU_MIMUL.eval_mIoU(args.input_output_directory, args.manufacturer, target, mode, id)
    except Exception as e: 
        logging.error (f"Unexpected error while trying to execute evaluation module with input {id} and FastSAM mask generated in {mode} mode for target {target}. Error: {e}")


    return(persam_IoU, persam_Acc, persam_f_IoU, persam_f_Acc)

if __name__ == "__main__":
    try:
        parse_args()
        setup_logging()
        main()
    except Exception as e:
        logging.error(f"Critical error occured, pirolease terminated. Error: {e}")
        sys.exit()