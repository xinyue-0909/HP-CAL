import argparse
import math
import time

from RandLANet import Network, log_out
from sampler import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="2", help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    parser.add_argument('--sampler', type=str, default='T', choices=["random", "T"], help='sampler')
    parser.add_argument('--dataset', type=str, default='S3DIS', choices=["S3DIS", "semantic3d", "SemanticKITTI"])
    parser.add_argument('--round', type=int, default=2)
    parser.add_argument('--classbal', type=int, default=0, choices=[0,1,2])
    parser.add_argument('--distance', type=int, default=0, choices=[0,1])

    parser.add_argument('--edcd', type=int, default=0, choices=[0,1])

    parser.add_argument('--uncertainty_mode', type=str, default="WetSU", choices=["mean", "sum_weight", "WetSU"], help='the mode from pixel uncertainty to region uncertainty')
    parser.add_argument('--point_uncertainty_mode', type=str, default="sb", choices=["lc", "sb", "entropy"],
                        help='point uncertainty')

    parser.add_argument('--oracle_mode', type=str, default="NAIL", choices=["dominant", "part_do", "NAIL", "domi_prec4", "domi_prec3"],
                        help='the mode from pixel uncertainty to region uncertainty. domi_prec4 denotes it begins using NAIL labeling when round 4 ')

    parser.add_argument('--reg_strength', default=0.008, type=float,
                        help='regularization strength for the minimal partition')
    parser.add_argument('--threshold', default=0.9, type=float,
                        help='tolerance threshold')
    parser.add_argument('--min_size', default=1, type=int,
                        help='the number of points in one selected superpoint >= min_size')

    parser.add_argument('--t', default=0, type=int,
                        help='t, multiple run')

    parser.add_argument('--gcn', default=0, type=int,
                        help='0: dont use gcn; 1: use gcn')

    parser.add_argument('--gcn_fps', default=0, type=int,
                        help='0: dont use gcn_fps; 1: use gcn_fps')

    parser.add_argument('--gcn_number', default=1, type=int,
                        help='convolution layers')

    parser.add_argument('--gcn_top', default=0, type=int,
                        help='0: default A; k: only convolution with top k neighbor nodes.')


    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    sampler_name = FLAGS.sampler
    dataset_name = FLAGS.dataset
    test_area = FLAGS.test_area
    round_num = FLAGS.round
    classbal = FLAGS.classbal
    distance = FLAGS.distance
    uncertainty_mode = FLAGS.uncertainty_mode
    oracle_mode = FLAGS.oracle_mode
    t = "t"+str(FLAGS.t)
    gcn = FLAGS.gcn
    gcn_fps = FLAGS.gcn_fps

    reg_strength = FLAGS.reg_strength
    point_uncertainty_mode = FLAGS.point_uncertainty_mode

    threshold = FLAGS.threshold
    min_size = FLAGS.min_size
    edcd = FLAGS.edcd
    gcn_number = FLAGS.gcn_number
    gcn_top = FLAGS.gcn_top

    if round_num >= 2:
        if dataset_name == "S3DIS":
            input_ = "input_0.040"
            test_area = 5
            cfg = ConfigS3DIS

        with open(os.path.join("./dataset/", dataset_name, str(reg_strength), "superpoint/total.pkl"), "rb") as f:
            total_obj = pickle.load(f)
        total_sp_num = total_obj["sp_num"]

        print("total_sp_num", total_sp_num)

        sampler_args = []

        if sampler_name == "random":
            sampler_args.append(t)
            sampler_args.append(sampler_name)
            sampler_args.append(oracle_mode)
            sampler_args.append(str(threshold))
            sampler_args.append(str(min_size))
            sampler_args.append(str(gcn_number))
            sampler_args.append(str(gcn_top))

            Sampler = RandomSampler(input_path="./dataset/" + dataset_name + "/" + input_, data_path="./dataset/" + dataset_name+ "/" + str(reg_strength),
                                    total_num=total_sp_num, sampler_args=sampler_args, min_size=min_size)

        elif sampler_name == "T":
            sampler_args.append(t)
            sampler_args.append(point_uncertainty_mode)

            if classbal == 1:
                sampler_args.append("classbal")
            elif classbal == 2:
                sampler_args.append("clsbal")
          
            if edcd == 1:
                sampler_args.append("edcd")
            if gcn:
                sampler_args.append("gcn")
            if gcn_fps:
                sampler_args.append("gcn_fps")

            sampler_args.append(uncertainty_mode)
            sampler_args.append(oracle_mode)
            sampler_args.append(str(threshold))
            sampler_args.append(str(min_size))
            sampler_args.append(str(gcn_number))
            sampler_args.append(str(gcn_top))

            Sampler = TSampler(input_path="./dataset/" + dataset_name + "/" + input_, data_path="./dataset/" + dataset_name+ "/" + str(reg_strength), total_num=total_sp_num,
                               test_area_idx=test_area, sampler_args=sampler_args, reg_strength=reg_strength, min_size=min_size, dataset_name=dataset_name)

        round_result_file = open(os.path.join("record_round", dataset_name + "_" + str(test_area) + "_" + get_sampler_args_str(sampler_args) + "_" + str(reg_strength) + '.txt'), 'a')

        sp_batch_size = 10000

        model = Network(cfg, dataset_name, sampler_args, test_area, reg_strength=reg_strength)
        model.restore_model(round_num=round_num - 1)
        for r in range(round_num, 6):
            begin_time = time.time()
            w = {"sp_num": 0, "p_num": 0, "sub_num": 0, "sub_p_num": 0, "ignore_sp_num": 0, "split_sp_num": 0}
            Sampler.sampling(model=model, batch_size=sp_batch_size, last_round=r-1, w=w, threshold=threshold,
                             gcn_gpu=1)

            labeling_region_num = w["sp_num"] + w["split_sp_num"]
            labeling_point_num = w["p_num"] + w["sub_p_num"]
            log_out("round= " + str(r) + " |          labeling mean point=" + str(labeling_point_num/labeling_region_num) + get_w(w) +", costTime=" + str(time.time()-begin_time), round_result_file)

            begin_time = time.time()
            best_miou, best_OA = model.train(round_num=r)

            log_out("round= " + str(r) + " | best_miou= " + str(best_miou) + ", best_OA= " + str(best_OA) +", costTime=" + str(time.time()-begin_time), round_result_file)

        model.close()
        round_result_file.close()
    else:
        print("round_num must >= 2", "round_num="+str(round_num))

