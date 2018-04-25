import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GTSRB_PATH = os.path.join(BASE_DIR, 'GTSRB', 'Final_Training', 'Images')

def create_gtsrb_labels():
    folders = [folder for folder in os.listdir(GTSRB_PATH) if os.path.isdir(os.path.join(GTSRB_PATH, folder))]
    images = []

    for folder in folders:
        print("Parsing folder: {}".format(folder))
        image_folder_path = os.path.join(GTSRB_PATH, folder)
        gt = os.path.join(image_folder_path, 'GT-'+folder+'.csv')
        with open(gt, 'r') as fp:
            reader = csv.reader(fp, delimiter=';')
            for idx, c in enumerate(reader):
                if not idx == 0:
                    images.append(image_folder_path+'/'+c[0]+','+c[3]+','+c[4]+','+c[5]+','+c[6]+','+c[7]+'\n')

    with open(os.path.join(GTSRB_PATH,"gt.csv"), "w") as fp:
        for line in images:
            fp.write(line)

create_gtsrb_labels()