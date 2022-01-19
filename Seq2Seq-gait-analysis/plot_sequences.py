import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import ipdb
import cmasher as cmr
from scipy import stats
from sklearn.metrics import precision_recall_curve, auc


INPUT_SEC_PREFIX = 'input_seq_step_'
PRED_SEC_PREFIX = 'prediction_at_output_step_'
COLOR_CODE = ['r','g','b']
create_plots = True
print_stat = True
plot_histogram = True

def plot_seq(npy_file):
    npy_file_pred = npy_file.replace(INPUT_SEC_PREFIX,PRED_SEC_PREFIX)
    seq_arr = np.load(npy_file)
    seq_arr_pred = np.load(npy_file_pred)
    #ipdb.set_trace()
    seq_to_plot = seq_arr[0,:,:]
    pred_to_plot = seq_arr_pred[0,:,:]
    for i in range(seq_arr_pred.shape[-1]):
        plt.plot(seq_to_plot[:,i],f'{COLOR_CODE[i]}--')
        plt.plot(pred_to_plot[:,i],f'{COLOR_CODE[i]}-')
    plt.title('seq')
    filename = npy_file.split('/')[-1].replace('.npy','')
    plt.savefig('{}'.format(filename))
    plt.close('all')

def print_distance_stat(npy_file):
    distance_arr = np.load(npy_file)
    filename = npy_file.split('/')[-1].replace('.npy','')
    print(f'{filename} - max_distance: {np.max(distance_arr)} min_distance: {np.min(distance_arr)} std_distance: {np.std(distance_arr)}')
    #plt.hist(distance_arr, bins = 50, alpha = 0.5, weights=np.ones(len(distance_arr)) / len(distance_arr))
    #plt.savefig('histogram_distances')
if create_plots:
    dir = '/home/dafnas1/gait_anomaly_detection/Seq2Seq-gait-analysis'
    for file in os.listdir(dir):
        if file.endswith(".npy"):
            if INPUT_SEC_PREFIX in file:
                plot_seq(os.path.join(dir,file))

if print_stat:
    print_distance_stat('/home/dafnas1/gait_anomaly_detection/Seq2Seq-gait-analysis/HD_without_preproc_euc_distance_arr.npy')
    print_distance_stat('/home/dafnas1/gait_anomaly_detection/Seq2Seq-gait-analysis/healthy_euc_distance_arr.npy')

if plot_histogram:
    hd_distance_arr = np.load('/home/dafnas1/gait_anomaly_detection/Seq2Seq-gait-analysis/HD_without_preproc_euc_distance_arr.npy')
    healthy_distance_arr = np.load('/home/dafnas1/gait_anomaly_detection/Seq2Seq-gait-analysis/healthy_euc_distance_arr.npy')

    plt.hist(hd_distance_arr, range = (0,10), bins=50, weights=np.ones(len(hd_distance_arr)) / len(hd_distance_arr)) 
    plt.hist(healthy_distance_arr, range=(0,10), bins=50, alpha=0.75, weights=np.ones(len(healthy_distance_arr)) / len(healthy_distance_arr))
    plt.legend(['hd_distance_arr','healthy_distance_arr'])
    plt.savefig('histogram_distances')
    plt.close('all')

    chorea_labels = np.load('/home/dafnas1/gait_anomaly_detection/Seq2Seq-gait-analysis/walking_HD_chorea_score_labels.npy')
    plt.hist(healthy_distance_arr, range=(0,10), bins=100, color = 'midnightblue', alpha=0.75, weights=np.ones(len(healthy_distance_arr)) / len(healthy_distance_arr))
    unique_labels = np.unique(chorea_labels)
    n_lines = len(unique_labels)
    # c = np.arange(1, n_lines + 1)
    # norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    # cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greys)
    

    #colors = plt.cm.Blues_r(np.linspace(0, 1, n_lines))
    colors = cmr.take_cmap_colors('rainbow_r', len(unique_labels[:-1]))
    #ipdb.set_trace()
    for i in unique_labels[:-1]:
        data = hd_distance_arr[np.where(chorea_labels == i)[0]]
        plt.hist(data, range=(0,10), bins=100, color = colors[i] ,alpha=0.75, weights=np.ones(len(data)) / len(data))
    legend = ['healthy'] + [f'HD_level_{i}' for i in unique_labels]
    plt.ylim(top=0.1)
    plt.legend(legend)
    plt.savefig('histogram_distance_per_chorea_level')
    plt.close('all')
    distance_chorea_dict={}
    for i in unique_labels[:-1]:
        distance_chorea_dict['{} level'.format(i)] = hd_distance_arr[np.where(chorea_labels == i)[0]]
    
    
    """ t-tests"""

    print('0 level arr - len: {:.2f} min: {:.2f} max= {:.2f} mean: {:.2f} std: {:.2f}'.format(len(distance_chorea_dict['0 level']),np.min(distance_chorea_dict['0 level']),
        np.max(distance_chorea_dict['0 level']) ,np.mean(distance_chorea_dict['0 level']), np.std(distance_chorea_dict['0 level'])))
    print('2 level arr - len: {:.2f} min: {:.2f} max= {:.2f} mean: {:.2f} std: {:.2f}'.format(len(distance_chorea_dict['2 level']),np.min(distance_chorea_dict['2 level']),
        np.max(distance_chorea_dict['2 level']),np.mean(distance_chorea_dict['2 level']), np.std(distance_chorea_dict['2 level'])))


    t_value,p_value=stats.ttest_ind(distance_chorea_dict['0 level'],distance_chorea_dict['2 level'], axis=0)

    print('Test statistic is %f'%float("{:.6f}".format(t_value)))
    print('p-value for two tailed test is %f'%p_value)

    ''' presicion recall curve '''
    auc_list = []
    for hd_level in range(1,4):
        pred = np.concatenate([distance_chorea_dict['0 level'],distance_chorea_dict[f'{hd_level} level']])
        len_0 = len(distance_chorea_dict['0 level'])
        labels = [index>=len_0 for index in range(len(pred))]
        precision, recall, thresholds = precision_recall_curve(labels, pred)
        plt.plot(recall,precision,color = colors[hd_level])
        auc_score = auc(recall, precision)
        auc_list.append(auc_score)

    #ipdb.set_trace()    
    plt.legend([f'{hd_level} level' for hd_level in range(1,4)])
    plt.savefig('recall_precision_curve')
    
    for hd_level in range(1,4):
            print('auc score for chorea level {}:{:.2f}'.format(hd_level,auc_list[hd_level-1]))
    print('mean auc score:{:.2f}'.format(np.mean(auc_list)))



