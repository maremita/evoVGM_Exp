import sys
import os.path

from joblib import dump, load

__author__ = "amine"


if __name__ == '__main__':

    job_code = sys.argv[1]
    verbose = 1

    print("Uncompressing {} experiments\n".format(job_code))

    model_types = [
            "jc69",
            "k80", 
            "gtr"
            ]

    data_types = [
            "jc69",
            "k80", 
            "gtr"
            ]

    branches = [
            3,
            4,
            5,
            ]

    len_alns = [
            100, 
            1000,
            5000,
            #10000
            ]

    output_dir = "../exp_outputs/{}/".format(job_code)

    for data_model in data_types:
        for nb_seqs in branches:
            for len_aln in len_alns:
                for evomodel in model_types:
 
                    # nb3_l5k_datajc69_evojc69.ini
                    exp_name = "nb{}_l{}_data{}_evo{}".format(
                            nb_seqs, len_aln, data_model, evomodel)

                    output_exp = output_dir+"{}/{}/".format(
                            evomodel, exp_name)

                    results_file = output_exp+\
                            "{}_results.pkl".format(exp_name)

                    uncompress_file = output_exp+\
                            "{}_results_uncompress.pkl".format(
                                    exp_name)

                    if not os.path.isfile(results_file):
                        print("{} doesn't exist!\n".format(
                            results_file))
                    else:
                        if verbose:
                            print(exp_name)
                        result_data = load(results_file)
                        dump(result_data, uncompress_file)
