import numpy as np
import os
import wfdb     # had to pip install
from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
import shutil
import zipfile


def resample_unequal(ts, fs_in, fs_out):
    """
    interploration
    """
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_out == fs_in:
        return ts
    else:
        x_old = np.linspace(0, 1, num=fs_in, endpoint=True)
        x_new = np.linspace(0, 1, num=fs_out, endpoint=True)
        y_old = ts
        f = interp1d(x_old, y_old, kind='linear')
        y_new = f(x_new)
        return y_new

def preprocess_signals(all_record_name, labels_to_use, path, save_path):
    """
    Convert ECG signals into numpy objects, where each observation is a 1-second heartbeat
    """
    all_pid = []
    all_data = []
    all_label = []

    valid_lead = ['MLII'] 
    fs_out = 360

    for record_name in all_record_name:
        try:
            tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'atr').__dict__
            tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
        except:
            print('read data failed')
            continue
        fs = tmp_data_res[1]['fs']
        ## total 1 second for each
        left_offset = int(1.0*fs / 2)
        right_offset = int(fs) - int(1.0*fs / 2)

        lead_in_data = tmp_data_res[1]['sig_name']
        my_lead_all = []
        for tmp_lead in valid_lead:
            if tmp_lead in lead_in_data:
                my_lead_all.append(tmp_lead)
        if len(my_lead_all) != 0:
            for my_lead in my_lead_all:
                channel = lead_in_data.index(my_lead)
                tmp_data = tmp_data_res[0][:, channel]

                idx_list = list(tmp_ann_res['sample'])
                label_list = tmp_ann_res['symbol']
                for i in range(len(label_list)):
                    s = label_list[i]
                    if s in labels_to_use:
                        idx_start = idx_list[i]-left_offset
                        idx_end = idx_list[i]+right_offset
                        if idx_start < 0 or idx_end > len(tmp_data):
                            continue
                        else:
                            all_pid.append(record_name)
                            all_data.append(resample_unequal(tmp_data[idx_start:idx_end], fs, fs_out))
                            all_label.append(s)
                    else:
                        continue
                      
                print('record_name:{}, lead:{}, fs:{}, cumcount: {}'.format(record_name, my_lead, fs, len(all_pid)))
        else:
            print('lead in data: [{0}]. no valid lead in {1}'.format(lead_in_data, record_name))
            continue

    all_pid = np.array(all_pid)
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    print(all_data.shape)
    print(Counter(all_label))

    # np.save(os.path.join(save_path, 'mitdb_data.npy'), all_data)
    np.save(os.path.join(save_path, 'mitdb_label.npy'), all_label)
    np.save(os.path.join(save_path, 'mitdb_pid.npy'), all_pid)

    return all_data, all_label, all_pid

def convert_to_image(signal_data):
    """
    Convert numpy signal data into 2D grayscale images
    """
    path = 'data/2d-images/'
    if not os.path.isdir(path):
        os.mkdir(path)
        
    for count, i in enumerate(signal_data):
        fig = plt.figure(frameon=False)
        plt.plot(i) 
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        filename = path + str(count)+'.png'
        fig.savefig(filename, dpi=100)
        plt.close()

        img = Image.open(filename)
        imgGray = img.convert('L')
        imgGray.save(filename)
        img.close()
        imgGray.close()

        if count % 500 == 0:
            print(count)

    shutil.make_archive(path, 'zip', path)

    return


if __name__ == "__main__":

    path = 'data/mit-bih-arrhythmia-database-1.0.0'
    save_path = 'data/'
    labels_to_use = ['N', 'L', 'R', 'V', '/', 'A', 'F', 'f', 'j', 'a', 'E', 'J', 'e', 'Q', 'S']

    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')

    signal_data, labels, pids = preprocess_signals(all_record_name, labels_to_use, path, save_path)

    convert_to_image(signal_data)