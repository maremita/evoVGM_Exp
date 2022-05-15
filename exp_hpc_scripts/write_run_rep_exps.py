#!/usr/bin/env python

import sys
import os
import os.path
from os import makedirs
import configparser
from datetime import datetime

__author__ = "amine"


if __name__ == '__main__':

    job_code = None
    sb_program = "submit_evo_exps.sh"
    program = "eval_run_replicates.py"
 
    if len(sys.argv) < 2:
        print("Config file is missing!!")
        sys.exit()

    if len(sys.argv) == 3:
        job_code = sys.argv[2]

    if job_code:
        str_time = job_code
        scores_from_file = "True"
    else:
        now = datetime.now()
        str_time = now.strftime("%m%d%H%M")
        scores_from_file = "False"

    print("Runing {} experiments\n".format(str_time))

    max_iter = "10000"
    n_reps = "10"

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
            5000,
            #10000
            ]

    exec_time = "24:00:00"
    mem = "80000M"
    cpus_per_task = "12"

    # For testing
    #exec_time = "00:05:00"
    #mem = "8000M"

    ## Fetch argument values from ini file
    ## ###################################
    config_file = sys.argv[1]
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    
    with open(config_file, "r") as cf:
        config.read_file(cf)

    config_path = "../exp_configs/{}/".format(str_time)
    makedirs(config_path, mode=0o700, exist_ok=True)

    output_dir = "../exp_outputs/{}/".format(str_time)
    makedirs(output_dir, mode=0o700, exist_ok=True)

    config.set("io", "output_path", output_dir)
    config.set("io", "scores_from_file", scores_from_file)
    
    config.set("hperparams", "n_epochs", max_iter)
    config.set("hperparams", "nb_replicates", n_reps)

    job_dir = "../exp_jobs/{}/".format(str_time)
    makedirs(job_dir, mode=0o700, exist_ok=True)

    for data_model, rates, freqs in data_types:
        for nb_seqs, branch_lens in branches:
            for len_aln in len_alns:
                for evomodel in model_types:
            
                    # nb3_l5k_datajc69_evojc69.ini
                    exp_name = "nb{}_l{}_data{}_evo{}".format(nb_seqs, 
                        len_aln, data_model, evomodel)

                    # Update configs
                    config.set("data", "alignment_size", str(len_aln))
                    config.set("data", "branch_lengths", branch_lens) 
                    config.set("data", "rates", rates)
                    config.set("data", "freqs", freqs)
                    config.set("subvmodel", "evomodel", evomodel)
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
