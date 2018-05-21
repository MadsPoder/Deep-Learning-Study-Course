import os

#Paths GTSDB
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, 'datasets', 'GTSDB', 'FullIJCNN2013')
SNAPSHOTS_DIR = os.path.join(BASE_DIR, 'snapshots')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

GROUND_TRUTH = os.path.join(DATASET_DIR, "gt.txt")
TRAIN_GROUND_TRUTH = os.path.join(DATASET_DIR, "gt_train.csv")
TEST_GROUND_TRUTH = os.path.join(DATASET_DIR, "gt_test.csv")
CLASS_MAPPINGS = os.path.join(DATASET_DIR, "cm.csv")

TRAIN_GROUND_TRUTH_COMBINED = os.path.join(DATASET_DIR, "gt_train_combined.csv")
TEST_GROUND_TRUTH_COMBINED = os.path.join(DATASET_DIR, "gt_test_combined.csv")
CLASS_MAPPINGS_COMBINED = os.path.join(DATASET_DIR, "cm_combined.csv")