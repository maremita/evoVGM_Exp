#!/usr/bin/env python

import sys
import os
import os.path
from os import makedirs
import configparser
from datetime import datetime
import json

__author__ = "amine"

"""
python write_run_rep_exps.py ../exp_2022_bcb/evovgm_reps_config.ini\
        job_code
"""

if __name__ == '__main__':

    sb_program = "slurm_exps.sh"
    program = "evovgm.py"
    job_code = None
 
    if len(sys.argv) < 2:
        print("Config files is missing!!")
        sys.exit()

    # Configuration ini file
    config_file = sys.argv[1]

    if len(sys.argv) == 3:
        job_code = sys.argv[2]

    if job_code:
        scores_from_file = "True"
    else:
        now = datetime.now()
        job_code = now.strftime("%m%d%H%M")
        scores_from_file = "False"

    print("Runing {} experiments...\n".format(job_code))

    ## Fetch argument values from ini file
    ## ###################################
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    
    with open(config_file, "r") as cf:
        config.read_file(cf)
    
    # Get write_run_rep_exps.py specific sections:
    if config.has_section("slurm"):
        run_slurm = config.getboolean("slurm", "run_slurm")
        # if run_slurm=False: the script will launch the jobs locally
        # elif run_slurm=True: it will launch the jobs on slurm

        # SLURM parameters
        exec_time = config.get("slurm", "exec_time")
        mem = config.get("slurm", "mem")
        cpus_per_task = config.get("slurm", "cpus_per_task")
    else:
        # The script will launch the jobs locally;
        run_slurm = False

    run_jobs = config.getboolean("evaluation", "run_jobs")
    #if run_jobs=False: the script generates the evovgm config files 
    #but it won't launch the jobs. If run_jobs=True: it runs the jobs

    max_iter = config.get("evaluation", "max_iter") # needs to be str
    n_reps = config.get("evaluation", "n_reps") # needs to be str

    model_types = json.loads(config.get("evaluation", "model_types"))
    data_types = json.loads(config.get("evaluation", "data_types"))
    branches = json.loads(config.get("evaluation", "branches"))
    len_alns = json.loads(config.get("evaluation", "len_alns"))

    ## Remove slurm and evaluation sections from the config object
    ## to not include them in the config files of evovgm.py
    if config.has_section("slurm"):
        config.remove_section('slurm')
    config.remove_section('evaluation')

    ## 
    config_path = "../exp_configs/{}/".format(job_code)
    makedirs(config_path, mode=0o700, exist_ok=True)

    output_dir = "../exp_outputs/{}/".format(job_code)
    makedirs(output_dir, mode=0o700, exist_ok=True)

    config.set("io", "output_path", output_dir)
    config.set("io", "scores_from_file", scores_from_file)
    
    config.set("hperparams", "n_epochs", max_iter)
    config.set("hperparams", "nb_replicates", n_reps)

    job_dir = "../exp_jobs/{}/".format(job_code)
    makedirs(job_dir, mode=0o700, exist_ok=True)

    for data_model, rates, freqs in data_types:
        for nb_seqs, branch_lens in branches:
            for len_aln in len_alns:
                for evomodel in model_types:
            
                    # nb3_l5k_datajc69_evojc69.ini
                    exp_name = "nb{}_l{}_data{}_evo{}".format(
                            nb_seqs, len_aln, data_model, evomodel)

                    # Update configs
                    config.set("data","alignment_size", str(len_aln))
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
 
                    if run_slurm:
                        s_error = os.path.join(job_dir,
                                exp_name+"_%j.err")
                        s_output = os.path.join(job_dir, 
                                exp_name+"_%j.out")

                        cmd = "sbatch --job-name={} --time={}"\
                                " --export=PROGRAM={},CONF_file={}"\
                                " --mem={} --cpus-per-task={}"\
                                " --error={} --output={} {}".format(
                                        exp_name,
                                        exec_time, 
                                        program,
                                        config_file,
                                        mem,
                                        cpus_per_task,
                                        s_error,
                                        s_output,
                                        sb_program)
                    else:
                        cmd = "{} {} &".format(program, config_file)

                    res_file = output_dir+\
                            "{}/{}/{}_results.pkl".format(
                                    evomodel, exp_name, exp_name)

                    if not os.path.isfile(res_file):
                        print("\n", exp_name)
                        if run_jobs:
                            print(cmd)
                            #os.system(cmd)
