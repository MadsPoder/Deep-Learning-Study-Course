import os
from PIL import Image

#Paths GTSDB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_PATH = os.path.join(BASE_DIR, 'FullIJCNN2013')
CLEANED_PATH = os.path.join(BASE_DIR, 'FullIJCNN2013_converted')
GROUND_TRUTH = os.path.join(FILES_PATH,"gt.txt")
GROUND_TRUTH_CONVERTERD = os.path.join(CLEANED_PATH,"gt.csv")
CLASS_MAPPINGS = os.path.join(CLEANED_PATH,"cm.csv")

def convert_gtsdb_images():
    print("--------- Converting GTSDB Images ---------")
    files = [file for file in os.listdir(FILES_PATH) if os.path.splitext(file)[1] == ".ppm"]

    for idx, file in enumerate(files):
        print("Converting image {} of {}".format(idx, len(files)-1), end='\r')
        #Converting from ppm to jpg using pillow https://stackoverflow.com/a/27046228
        im = Image.open(os.path.join(FILES_PATH, file))
        im.save(os.path.join(CLEANED_PATH, os.path.splitext(file)[0]+".png"))

def convert_gtsdb_labels():
    print("--------- Converting GTSDB Labels ---------")
    parsed = []

    #Ground truth conversion
    with open(GROUND_TRUTH, "r") as fp:
        for line in fp:
            parsed.append(CLEANED_PATH+"/"+line.replace(";",",").replace("ppm","png"))
    
    with open(GROUND_TRUTH_CONVERTERD, "w") as fp:
        for line in parsed:
            fp.write(line)
    
    #Class mappings (for now just numbers)
    with open(CLASS_MAPPINGS, "w") as fp:
        for number in list(range(43)):
            fp.write(str(number)+","+str(number)+"\n")

if not os.path.exists(CLEANED_PATH):
    os.makedirs(CLEANED_PATH)

#convert_images()
#create_labels()
