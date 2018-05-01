import os, random, math

#Paths GTSDB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets', 'GTSDB', 'FullIJCNN2013')
GROUND_TRUTH = os.path.join(DATASET_DIR, "gt.txt")
TRAIN_GROUND_TRUTH = os.path.join(DATASET_DIR, "gt_train.csv")
TEST_GROUND_TRUTH = os.path.join(DATASET_DIR, "gt_test.csv")
CLASS_MAPPINGS = os.path.join(DATASET_DIR, "cm.csv")

def convert_gtsdb_labels():
    random.seed(42)
    print("--------- Splitting GTSDB into train/test ---------")

    parsed, train, test = [], [], []

    #Ground truth conversion
    with open(GROUND_TRUTH, "r") as fp:
        for line in fp:
            #parsed.add(DATASET_DIR+"/"+line.replace(";",","))
            parsed.append(DATASET_DIR+"/"+line.replace(";",","))

    train = random.sample(parsed, math.ceil(0.7*len(parsed)))
    test = list(set(parsed) - set(train))

    with open(TRAIN_GROUND_TRUTH, "w") as fp:
        for line in train:
            fp.write(line)
    
    with open(TEST_GROUND_TRUTH, "w") as fp:
        for line in test:
            fp.write(line)

    #Class mappings (for now just numbers)
    with open(CLASS_MAPPINGS, "w") as fp:
        for number in list(range(43)):
            fp.write(str(number)+","+str(number)+"\n")

convert_gtsdb_labels()