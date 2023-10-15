import tensorflow as tf
from model import create_model
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import csv
import time
import argparse
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

drv.init()
pycuda_ctx = drv.Device(0).retain_primary_context()
pycuda_ctx.push()

# set a seed for reproducibility
np.random.seed(42)
tf.compat.v1.set_random_seed(42)

total_emb = 0
total_vot = 0
total_cos = 0

total_load = 0
total_thresh = 0

# if you don't use GPU, comment out the following
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

parser = argparse.ArgumentParser ()
parser.add_argument ('--test', default='./data/datasets/test_data/2094_test.npz')
parser.add_argument ('--flow', default=2094)
parser.add_argument ('--tor_len', default=500)
parser.add_argument ('--exit_len', default=800)
parser.add_argument ('--model1', default='./data/model/crawle_overlap_new2021_11_interval5_addn2_model1_w_superpkt_0.0033')
parser.add_argument ('--model2', default='./data/model/crawle_overlap_new2021_11_interval5_addn2_model2_w_superpkt_0.0033')
parser.add_argument ('--output', default="./data/results/output.csv")
args = parser.parse_args ()


def get_session(gpu_fraction=0.85):
    gpu_options = tf.compat.v1.GPUOptions (per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    return tf.compat.v1.Session (config=tf.compat.v1.ConfigProto (gpu_options=gpu_options))


def Cosine_Similarity_eval(tor_embs, exit_embs, single_output_l, correlated_shreshold, muti_output_list):

    global total_vot
    start_vot = time.time ()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for tor_eval_index in range (0, tor_embs.shape[0]):
        for exit_eval_index in range (0, tor_embs.shape[0]):
            cos_condithon_a = (tor_eval_index == exit_eval_index) 
            number_of_ones = (single_output_l[(tor_eval_index * (tor_embs.shape[0])) + exit_eval_index])
            cos_condition_b = (number_of_ones >= correlated_shreshold)
            cos_condition_c = (number_of_ones < correlated_shreshold)

            if (cos_condithon_a and cos_condition_b):
                TP = TP + 1
            if (cos_condithon_a and cos_condition_c):
                FN = FN + 1
            if ((not (cos_condithon_a)) and cos_condition_b):
                FP = FP + 1
            if ((not (cos_condithon_a)) and cos_condition_c):
                TN = TN + 1

    if ((TP + FN) != 0):
        TPR = (float) (TP) / (TP + FN)
    else:
        TPR = -1

    if ((FP + TN) != 0):
        FPR = (float) (FP) / (FP + TN)
    else:
        FPR = -1

    muti_output_list.append (TPR)
    muti_output_list.append (FPR)
    muti_output_list.append(calculate_bdr(TPR, FPR))

    end_time = time.time ()
    total_vot = total_vot + (end_time - start_vot)


def calculate_bdr(tpr, fpr):
    TPR = tpr
    FPR = fpr
    c = 1 / int(args.flow)
    u = (int(args.flow)-1) / int(args.flow)
    if ((TPR * c) + (FPR * u)) != 0:
        BDR = (TPR * c) / ((TPR * c) + (FPR * u))
    else:
        BDR = -1
    return BDR


def preprocessing_new_test_data(win, number_of_interval):
    npz_path = '../data/obfs_new/obfs4_new_interval' + str (number_of_interval) + '_win' + str (win) + '.npz'
    np_array = np.load (npz_path, encoding='latin1', allow_pickle=True)
    tor_seq = np_array["tor"]
    exit_seq = np_array["exit"]
    number_of_traces = tor_seq.shape[0]
    print (number_of_traces)
    print (type (tor_seq[0]))
    print (len (tor_seq[0]))
    print (tor_seq[0][1])
    '''
    [ [{'ipd': x, 'size': y},{}.....{}]
      [{},{}.....{}]
      [{},{}.....{}] ]
    '''

    for i in range (0, number_of_traces):
        tor_seq[i] = [float (pair["ipd"]) * 1000.0 for pair in tor_seq[i]] + [float (pair["size"]) / 1000.0 for pair in
                                                                              tor_seq[i]]
        if len (tor_seq[i]) < (500 * 2):
            tor_seq[i] = tor_seq[i] + ([0] * ((500 * 2) - (len (tor_seq[i]))))
        elif len (tor_seq[i]) > (500 * 2):
            tor_seq[i] = tor_seq[i][0:(500 * 2)]

        exit_seq[i] = [float (pair["ipd"]) * 1000.0 for pair in exit_seq[i]] + [float (pair["size"]) / 1000.0 for pair
                                                                                in exit_seq[i]]
        if len (exit_seq[i]) < (800 * 2):
            exit_seq[i] = exit_seq[i] + ([0] * ((800 * 2) - (len (exit_seq[i]))))
        elif len (exit_seq[i]) > (800 * 2):
            exit_seq[i] = exit_seq[i][0:(800 * 2)]

    tor_test = np.reshape (np.array (list (tor_seq)), (2094, 1000, 1))
    exit_test = np.reshape (np.array (list (exit_seq)), (2094, 1600, 1))
    print (tor_test[0][1])
    return (tor_test, exit_test)



############################ CUDA Code #################################################

mod = SourceModule("""

extern "C" {

__global__ void threshold_finder_2094(float *output_threshold_list_gpu, float *input_similarity, float *temp, int *thres_seed)
{
    int tid = threadIdx.x;
    int cut_point[698];
    int i;

        for (i = 0; i < 3; i++) {
            cut_point[tid] = (sizeof(input_similarity[2094]) - 1) * (thres_seed[0]/100);
            output_threshold_list_gpu[tid + i*698] = temp[cut_point[tid] + tid*2094 + i*698*2094];
        }
    }

__global__ void vote_2094(float *single_output, float *similarity_threshold, float *input_similarity_list)
{
    int tid = threadIdx.x;
    int num_of_lines = 3;
    int i,t, j, constant_num;

        for (i = 0; i < num_of_lines; i++) {
            t = similarity_threshold[tid + i*2094];
            constant_num = i*num_of_lines*698*698 + tid*num_of_lines*698;
            for (j = 0; j < num_of_lines; j++) {
                if (input_similarity_list[i*tid + j*tid] >= t)
                    single_output[constant_num + j*tid] = single_output[constant_num + j*tid] + 1;
            }   
        }
    }


__global__ void threshold_finder_5k(float *output_threshold_list_gpu, float *input_similarity, float *temp, int *thres_seed)
{
    int tid = threadIdx.x;
    int cut_point[1000];
    int i;

        for (i = 0; i < 5; i++) {
            cut_point[tid] = (sizeof(input_similarity[5000]) - 1) * (thres_seed[0]/100);

            output_threshold_list_gpu[tid + i*1000] = temp[cut_point[tid] + tid*5000 + i*1000*5000];
        }
    }

__global__ void vote_5k(float *single_output, float *similarity_threshold, float *input_similarity_list)
{
    int tid = threadIdx.x;
    int num_of_lines = 5;
    int i,t, j, constant_num;

        for (i = 0; i < num_of_lines; i++) {
            t = similarity_threshold[tid + i*1000];
            constant_num = i*num_of_lines*1000*1000 + tid*num_of_lines*1000;
            for (j = 0; j < num_of_lines; j++) {
                if (input_similarity_list[i*tid + j*tid] >= t)
                    single_output[constant_num + j*tid] = single_output[constant_num + j*tid] + 1;
            }   
        }
    }

__global__ void threshold_finder_7500(float *output_threshold_list_gpu, float *input_similarity, float *temp, int *thres_seed)
{
    int tid = threadIdx.x;
    int cut_point[750];
    int i;

        for (i = 0; i < 10; i++) {
            cut_point[tid] = (sizeof(input_similarity[7500]) - 1) * (thres_seed[0]/100);
            output_threshold_list_gpu[tid + i*750] = temp[cut_point[tid] + tid*7500 + i*750*7500];
        }
    }

__global__ void vote_7500(float *single_output, float *similarity_threshold, float *input_similarity_list)
{
    int tid = threadIdx.x;
    int num_of_lines = 10;
    int i,t, j, constant_num;

        for (i = 0; i < num_of_lines; i++) {
            t = similarity_threshold[tid + i*750];
            constant_num = i*num_of_lines*750*750 + tid*num_of_lines*750;
            for (j = 0; j < num_of_lines; j++) {
                if (input_similarity_list[i*tid + j*tid] >= t)
                    single_output[constant_num + j*tid] = single_output[constant_num + j*tid] + 1;
            }   
        }
    }

__global__ void threshold_finder_10k(float *output_threshold_list_gpu, float *input_similarity, float *temp, int *thres_seed)
{
    int tid = threadIdx.x;
    int cut_point[1000];
    int i;

        for (i = 0; i < 10; i++) {
            cut_point[tid] = (sizeof(input_similarity[10000]) - 1) * (thres_seed[0]/100);
            output_threshold_list_gpu[tid + i*1000] = temp[cut_point[tid] + tid*10000 + i*1000*10000];
        }
    }

__global__ void vote_10k(float *single_output, float *similarity_threshold, float *input_similarity_list)
{
    int tid = threadIdx.x;
    int num_of_lines = 10;
    int i,t, j, constant_num;

        for (i = 0; i < num_of_lines; i++) {
            t = similarity_threshold[tid + i*10000];
            constant_num = i*num_of_lines*10000*10000 + tid*num_of_lines*1000;
            for (j = 0; j < num_of_lines; j++) {
                if (input_similarity_list[i*tid + j*tid] >= t)
                    single_output[constant_num + j*tid] = single_output[constant_num + j*tid] + 1;
            }   
        }
    }
                   

}
""")


def eval_model(full_or_half, five_or_four, use_new_data, model1_path, model2_path, test_path, thr, use_global,
               muti_output_list, soft_muti_output_list):
    global total_emb
    global total_vot
    global total_cos

    global total_load
    global total_thresh

    load_t = time.time()
    test_data = np.load (test_path, allow_pickle=True)
    total_load = total_load + (time.time() - load_t)
    # print(test_data['tor'][0].shape)
    # print(test_data['exit'][0].shape)
    pad_t = int(args.tor_len)*2#500*2 #238*2
    pad_e = int(args.exit_len)*2#800*2 #100*2

    tor_model = create_model(input_shape=(pad_t, 1), emb_size=64, model_name='tor')
    exit_model = create_model(input_shape=(pad_e, 1), emb_size=64, model_name='exit')

    # load triplet models for tor and exit traffic
    tor_model.load_weights(model1_path + ".h5")
    exit_model.load_weights(model2_path + ".h5")

    cosine_similarity_table = []

    activated_windows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    last_activated_window = 10
    correlated_shreshold_value = five_or_four
    thres_seed = thr

    load_t = time.time()

    test_data_tor = test_data['tor']
    test_data_exit = test_data['exit']

    total_load = total_load + (time.time() - load_t)

    for win in range(11):
        print("~~~~~~~~~~ We are in window %d ~~~~~~~~~~~" % win)

        if use_new_data == 1:
            (test_data_tor, test_data_exit) = preprocessing_new_test_data(win, 5)
        elif use_new_data != 1:
            start_load_win = time.time()
            
            tor_win = test_data_tor[win]
            exit_win = test_data_exit[win]
            
            end_load_win = time.time()
            total_load = total_load + (end_load_win - start_load_win)

        start_emd = time.time ()
        tor_embs = tor_model.predict (tor_win, verbose=0)
        exit_embs = exit_model.predict (exit_win, verbose=0)
        end_emd = time.time ()

        total_emb = total_emb + (end_emd - start_emd)

        start_cos = time.time ()
        cosine_similarity_table = cosine_similarity (tor_embs, exit_embs)
        end_cos = time.time ()

        total_cos = total_cos + (end_cos - start_cos)

        if flow_length == 2094:
            threshold_finder = mod.get_function("threshold_finder_2094")
            vote = mod.get_function("vote_2094")
            block = (698, 1, 1)
        elif flow_length == 5000:
            threshold_finder = mod.get_function("threshold_finder_5k")
            vote = mod.get_function("vote_5k")
            block = (1000, 1, 1)
        elif flow_length == 7500:
            threshold_finder = mod.get_function("threshold_finder_7500")
            vote = mod.get_function("vote_7500")
            block = (750, 1, 1)
        elif flow_length == 10000:
            threshold_finder = mod.get_function("threshold_finder_10k")
            vote = mod.get_function("vote_10k")
            block = (1000, 1, 1)

        start_thresh_find = time.time ()
        
        input_similarity_list = np.array(cosine_similarity_table)
        thres_seed = np.array(thres_seed).astype(np.int32)
        
        temp = np.zeros_like(input_similarity_list)

        for i in range(temp.shape[0]):
            temp[i] = np.sort(input_similarity_list[i])[::-1]

        input_similarity_list = input_similarity_list.astype(np.float32).flatten()
        temp = temp.astype(np.float32).flatten()

        output_threshold_list_gpu = np.zeros(flow_length).astype(np.float32)

        threshold_finder(
            drv.Out(output_threshold_list_gpu), drv.In(input_similarity_list), drv.In(temp), drv.In(thres_seed),
            block=block, grid=(1, 1))
        
        end_thresh_find = time.time ()
        total_thresh = total_thresh + (end_thresh_find - start_thresh_find)

        start_vot = time.time ()

        threshold_list = np.array(output_threshold_list_gpu)
        single_output = np.zeros_like(input_similarity_list)  

        vote(
            drv.Out(single_output), drv.In(threshold_list), drv.In(input_similarity_list), block=block,
            grid=(1, 1))

        single_output = single_output.tolist()
        
        end_vot = time.time ()
        total_vot = total_vot + (end_vot - start_vot)


        if win == last_activated_window:
            Cosine_Similarity_eval(tor_embs, exit_embs, single_output, correlated_shreshold_value, muti_output_list)



if __name__ == "__main__":
    # if you don't use GPU, comment out the following
    tf.compat.v1.keras.backend.set_session (get_session ())
    start_time = time.time ()
    test_path = args.test
    model1_path = args.model1
    model2_path = args.model2

    # For time complexity analysis, use only one threshold (e.g., [60])
    # rank_thr_list = [60,50,47,43,40,37,33,28,24,20,16.667,14,12.5,11,10,9,8.333,7,6.25,5,4.545,3.846,2.941,1.667,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]#[10]
    rank_thr_list = [60]

    num_of_thr = len(rank_thr_list)

    flow_length = int(args.flow)

    five_or_four = 9

    rank_multi_output = []
    five_rank_multi_output = []
    for i in range (0, num_of_thr):
        rank_multi_output.append ([(rank_thr_list[i])])
        five_rank_multi_output.append ([(rank_thr_list[i])])

    epoch_index = 0
    use_global = 0
    use_new_data = 0

    for thr in rank_thr_list:
        eval_model(flow_length, five_or_four, use_new_data, model1_path, model2_path, test_path, thr, use_global,
                    rank_multi_output[epoch_index], [])
        epoch_index = epoch_index + 1
    
    pycuda_ctx.pop()
    
    end_time = time.time()

    print(f"-------- Flow: {flow_length} --------")
    print(f"total load time: {total_load}")
    print(f"total emb time: {total_emb}")
    print(f"total cos time: {total_cos}")
    print(f"total thresh time: {total_thresh}")
    print(f"total vote time: {total_vot}")

    # write the time complexity analysis results to a csv file
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["total_flows", "load_time", "emb_time", "cos_time", "thresh_time", "vote_time"])
        writer.writerow([flow_length, total_load, total_emb, total_cos, total_thresh, total_vot])