#!/usr/bin/env python

import sys
import os
import os.path
from os import makedirs
import configparser
from datetime import datetime

__author__ = "amine"

"""
python write_run_hyperpam_exps.py eval_conf_template.ini job_code
"""

if __name__ == '__main__':

    sb_program = "slurm_exps.sh"
    program = "evovgm.py"

    if len(sys.argv) < 2:
        print("Config file is missing!!")
        sys.exit()

    if len(sys.argv) == 3:
        job_code = sys.argv[2]

    # Configuration ini file
    config_file = sys.argv[1]

    if job_code:
        scores_from_file = "True"
    else:
        now = datetime.now()
        job_code = now.strftime("%m%d%H%M")
        scores_from_file = "False"

    print("Runing {} experiments...\n".format(job_code))

    max_iter = "5000"
    n_reps = "10"

    hyper_types = {
            "kl": ("alpha_kl", [0.0001, 0.001, 0.01, 0.1], 0.0001),
            "hs": ("hidden_size", [4, 16, 32, 64], 32),
            "ns": ("nb_samples", [1, 10, 100, 1000], 100),
                #[1000], 100),
            "lr": ("learning_rate",
                [0.00005, 0.0005, 0.005, 0.05], 0.005),
                #[0.05], 0.005),
            }

    model_types = [
            "jc69",
            "k80", 
            "gtr"
            ]

    data_types = [
            ("jc69", "0.160, 0.160, 0.160, 0.160, 0.160, 0.160", 
                "0.25, 0.25, 0.25, 0.25"), 
            ("k80" , "0.250, 0.125, 0.125, 0.125, 0.125, 0.250",
                "0.25, 0.25, 0.25, 0.25"), 
            ("gtr" , "0.160, 0.050, 0.160, 0.090, 0.300, 0.240",
                "0.10, 0.45, 0.30, 0.15")
            ]

    branches = [
            (3, "0.1,0.3,0.45"),
            (4, "0.1,0.3,0.45,0.15"),
            (5, "0.1,0.3,0.45,0.15,0.2")
            ]

    len_alns = [
            100, 
            1000,
            5000
            ]

    # Choose here type of data and model
    ind_model = 2
    evomodel = model_types[ind_model]
    data_model, rates, freqs = data_types[ind_model]
    #
    nb_seqs, branch_lens = branches[1]
    len_aln = len_alns[1]

    exec_time = "24:00:00"
    mem = "100000M"
    cpus_per_task = "24"

    # For testing
    #exec_time = "00:05:00"
    #mem = "8000M"

    ## Fetch argument values from ini file
    ## ###################################
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    
    with open(config_file, "r") as cf:
        config.read_file(cf)

    config_path = "../exp_configs/{}/".format(job_code)
    makedirs(config_path, mode=0o700, exist_ok=True)

    output_dir = "../exp_outputs/{}/".format(job_code)
    makedirs(output_dir, mode=0o700, exist_ok=True)

    config.set("io", "output_path", output_dir)
    config.set("io", "scores_from_file", scores_from_file)    

    config.set("data", "alignment_size", str(len_aln))
    config.set("data", "branch_lengths", branch_lens) 
    config.set("data", "rates", rates)
    config.set("data", "freqs", freqs)
    config.set("subvmodel", "evomodel", evomodel)
    config.set("hperparams", "n_epochs", max_iter)
    config.set("hperparams", "nb_replicates", n_reps)

    job_dir = "../exp_jobs/{}/".format(job_code)
    makedirs(job_dir, mode=0o700, exist_ok=True)

    for hyper_code in hyper_types:
        hyper_name = hyper_types[hyper_code][0]
        hyper_values = hyper_types[hyper_code][1]

        # REset default values for other hyperparms
        for code in hyper_types:
            name = hyper_types[code][0]
            config.set("hperparams", name, str(hyper_types[code][2]))

        for hyper_value in hyper_values:
            # evogtr_hs_4 (old names)
            #exp_name = "evo{}_{}_{}".format(evomodel,
            #        hyper_code, hyper_value)

            # nb3_l5k_datajc69_evojc69_hs4.ini
            exp_name = "nb{}_l{}_data{}_evo{}_{}{}".format(
                    nb_seqs, len_aln, data_model, evomodel,
                    hyper_code, hyper_value)

            # Update configs
            config.set("hperparams", hyper_name, str(hyper_value))
            config.set("settings", "job_name", exp_name)

            # write it on a file
            config_file = os.path.join(config_path, "{}.ini"\
                    "".format(exp_name))

            with open (config_file, "w") as fh:
                config.write(fh)

            s_error = os.path.join(job_dir, exp_name+"_%j.err")
            s_output = os.path.join(job_dir, exp_name+"_%j.out")

            cmd = "sbatch --job-name={} --time={}"\
                    " --export=PROGRAM={},CONF_file={} "\
                    "--mem={} --cpus-per-task={} --error={}"\
                    " --output={} {}".format(exp_name, exec_time, 
                            program, config_file, mem, cpus_per_task,
                            s_error, s_output, sb_program)
            res_file = output_dir+"{}/{}/{}_results.pkl".format(
                    evomodel, exp_name, exp_name)

            if not os.path.isfile(res_file):
                print("\n", exp_name)
                #print(cmd)
                #os.system(cmd)
