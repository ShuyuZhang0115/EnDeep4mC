from colorama import Fore
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from xgboost import XGBClassifier
from fs.encode1 import ENAC2, binary, NCP, EIIP, CKSNAP, ENAC, Kmer, NAC, PseEIIP, ANF, CKSNAP8, Kmer4, TNC, RCKmer5, \
    DNC
from fs.load_acc import TAC
from fs.load_pse import SCPseTNC, PCPseTNC, PseKNC, SCPseDNC, PseDNC
import numpy as np
import pandas as pd
from tools.tools import split_h

import os
def load_data(dataset):
    base_dir = '/your_path/EnDeep4mC/data/4mC'  # The absolute path where the dataset is located
    dir1 = dataset.split("_")[0]
    train_pos = os.path.join(base_dir, dataset, "train_pos.txt")
    train_neg = os.path.join(base_dir, dataset, "train_neg.txt")
    test_pos = os.path.join(base_dir, dataset, "test_pos.txt")
    test_neg = os.path.join(base_dir, dataset, "test_neg.txt")

    data_train_pos = pd.read_table(train_pos, header=None)
    data_train_neg = pd.read_table(train_neg, header=None)
    data_test_pos = pd.read_table(test_pos, header=None)
    data_test_neg = pd.read_table(test_neg, header=None)

    train_seq = data_train_pos.iloc[:, 0].values.tolist() + data_train_neg.iloc[:, 0].values.tolist()
    train_seq_id = [1] * data_train_pos.shape[0] + [0] * data_train_neg.shape[0]
    data_train = pd.DataFrame({"label": train_seq_id, "seq": train_seq})

    test_seq = data_test_pos.iloc[:, 0].values.tolist() + data_test_neg.iloc[:, 0].values.tolist()
    test_seq_id = [1] * data_test_pos.shape[0] + [0] * data_test_neg.shape[0]
    data_test = pd.DataFrame({"label": test_seq_id, "seq": test_seq})

    '''
    #Print the positive and negative sample ratios and check if the data is unbalanced
    pos_count_train = (data_train['label'] == 1).sum()
    neg_count_train = (data_train['label'] == 0).sum()
    print(f"The ratio of positive and negative samples in the training set: {pos_count_train / neg_count_train}")
    pos_count_test = (data_test['label'] == 1).sum()
    neg_count_test = (data_test['label'] == 0).sum()
    print(f"The ratio of positive and negative samples in the test set: {pos_count_test / neg_count_test}")
    '''

    return data_train, data_test

def read_fasta_data(file_path):
    """
    Read a single FASTA format data file and return a list of sequence contents.

    参数:
    file_path: The path of the FASTA format data file

    返回:
    sequences: A list containing all the sequence contents in the file
    """
    sequences = []
    with open(file_path, "r") as f:
        lines = ""
        for line in f.readlines():
            line = line.strip()
            if not line.startswith(">"):
                line = line.upper()
                sequences.append(line)
    return sequences

def format_ilearn(seq_df1, is_train):
    seq_df = seq_df1.copy()
    seq_df["is_train"] = [is_train] * seq_df.shape[0]
    seq_id = ["seq_"+str(i) for i in range(seq_df1.shape[0])]
    seq_df["seq_id"] = seq_id
    seq_df = seq_df.loc[:,["seq_id","seq","label","is_train"]]
    return seq_df.values.tolist()


def repair(encoding):
    '''
    Arguments:
    encoding -- the encoding matrix calculated by iLearns
    Return:
    data -- DataFrame，feature matrix with row sample  and column feature
    label -- list，
    record_feature_type
    '''
    encoding = pd.DataFrame(encoding)
    encoding = encoding.iloc[1:,1:]
    label = encoding.iloc[:,0].values.astype(int)
    data = encoding.iloc[:,1:]
    return data,label

def ml_code(seq_df1, is_train="training", feature_extract_method=None):
    #pd.options.display.max_columns = None
    '''
    Convert the sequence matrix into a feature matrix
    :param seq_df1: (seq_id,seq)
    :param is_train: training, or testing
    :return:
    '''
    fastas1 = format_ilearn(seq_df1, is_train)
    encodings_trains = []
    label_train = []
    record_feature_type = []
    if feature_extract_method is None:
        feature_extract_method = {"ENAC": ENAC2, "binary": binary, "NCP": NCP, "EIIP": EIIP, "Kmer4": Kmer4, "CKSNAP": CKSNAP8,
                                  "PseEIIP": PseEIIP, "TNC": TNC, "RCKmer5": RCKmer5, "SCPseTNC": SCPseTNC, "PCPseTNC": PCPseTNC,
                                  "ANF": ANF, "NAC": NAC, "TAC": TAC}
        #feature_extract_method = {
        #    "ENAC": ENAC2, "binary": binary, "NCP": NCP, "CKSNAP": CKSNAP8, "PseEIIP": PseEIIP, "PseKNC": PseKNC,
        #    }
    for fsm in feature_extract_method.keys():
        method1 = feature_extract_method[fsm]
        # For each feature encoding method, firstly carry out the encoding conversion
        encoded_result = method1(fastas1)
        d_tr = pd.DataFrame(encoded_result)
        data_train, label_train = repair(d_tr)
        encodings_trains.append(data_train)
        record_feature_type.extend([fsm] * data_train.shape[1])
        # Output the first few rows of data of the feature matrix after the conversion of the current feature coding method
        # print(f"{fsm}The shape of the feature matrix after encoding:", data_train.shape)
        # print(f"{fsm}The data of the first 4 rows of the encoded feature matrix:")
        # print(data_train[:4])
    encoding_TR = pd.concat(encodings_trains, axis=1, ignore_index=True)
    print("encoding_TR:", encoding_TR.shape)
    return encoding_TR.values, np.array(label_train), record_feature_type

def xgb_feature_selection(data_train, label_train):
    data_train2, data_dev,label_train2,label_dev = split_h(data_train, label_train,freq=0.1)
    xgb_model = XGBClassifier(eval_metric="logloss")
    xgb_model.fit(data_train2, label_train2)
    cc = pd.Series(xgb_model.feature_importances_)
    cc = cc.sort_values(ascending = False)
    fs1 = [i for i in cc.index]
    result1 = []
    for i in tqdm(range(1,len(fs1),10), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.RED, Fore.RESET)):
        model = XGBClassifier(eval_metric='logloss')
        model.fit(data_train2[:,fs1[:i]], label_train2)
        y_submission_xgb = model.predict_proba(data_dev[:,fs1[:i]])[:, 1]
        y_submission_acc = [(1 if x>0.5 else 0) for x in y_submission_xgb]
        acc1 = accuracy_score(label_dev, y_submission_acc)
        # auc1 = roc_auc_score(label_dev, y_submission_xgb)
        result1.append(acc1)
        # result1.append(auc1)
    index_best = result1.index(max(result1))*10+1
    print("choose",str(index_best),"features")
    return fs1[:index_best]

