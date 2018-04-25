import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_PATH = os.path.join(BASE_DIR, 'FullIJCNN2013')
CLEANED_PATH = os.path.join(BASE_DIR, 'FullIJCNN2013_converted')

def convert_images():
    print("Converting images")
    files = [f for f in os.listdir(FILES_PATH) if os.path.splitext(f)[1] == ".ppm"]
    
    for file in files:
        #Converting from ppm to jpg using pillow https://stackoverflow.com/a/27046228
        im = Image.open(os.path.join(FILES_PATH, file))
        im.save(os.path.join(CLEANED_PATH, os.path.splitext(file)[0]+".png"))

if not os.path.exists(CLEANED_PATH):
    os.makedirs(CLEANED_PATH)

def create_labels():
    print("Creating labels")
    parsed = []

    #Ground truth conversion
    with open(os.path.join(FILES_PATH,"gt.txt"), "r") as fp:
        for line in fp:
            parsed.append(CLEANED_PATH+"/"+line.replace(";",",").replace("ppm","png"))
    
    with open(os.path.join(CLEANED_PATH,"gt.csv"), "w") as fp:
        for line in parsed:
            fp.write(line)
    
    #Class mappings (for now just numbers)
    with open(os.path.join(CLEANED_PATH,"cm.csv"), "w") as fp:
        for number in list(range(43)):
            fp.write(str(number)+","+str(number)+"\n")


#convert_images()
#create_labels()