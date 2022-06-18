#!/usr/bin/env python

from evoVGM.utils import str2floats, fasta_to_list
from evoVGM.utils import str_to_values
from evoVGM.reports import plt_elbo_ll_kl_rep_figure
from evoVGM.reports import aggregate_estimate_values
from evoVGM.reports import plot_fit_estim_dist
from evoVGM.reports import plot_fit_estim_corr

import sys
import os.path
from os import makedirs
import configparser

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist

from joblib import dump, load

import matplotlib.pyplot as plt

__author__ = "amine"


def write_dict_df(dict_df, filename):
    with pd.option_context(
            'display.max_rows', None, 'display.max_columns', None):
        pd.options.display.float_format = '{:.3f}'.format # {:,.3f}
        with open(filename, "w") as fh:
            for code in dict_df:
                fh.write(code)
                fh.write("\n")
                #fh.write(dict_df[code].to_csv(float_format='%.3f'))
                #fh.write("\n")
                #fh.write("\n")
                df = dict_df[code]
                #if "pval" in df.columns.get_level_values(1):
                #    df = df.drop("pval", axis=1, level=1)
                fh.write(df.to_latex())
                fh.write("\n")
                fh.write(dict_df[code].to_string())
                fh.write("\n")
                fh.write("\n")
                fh.write("#" * 69)
                fh.write("\n")
                fh.write("\n")


def plot_elbos_data_wise(
        fit_elbos,
        model_types,
        out_file,
        print_xtick_every=200,
        usetex=False,
        title="Title"):

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    sizefont = 10

    data_models = list(fit_elbos.keys())
    nb_dm = len(data_models)

    f, axs = plt.subplots(1, nb_dm, figsize=(7*nb_dm, 5))

    plt.rcParams.update({'font.size':sizefont, 'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.15, hspace=0.1)

    # Add more colors if nb models > 3 or use a template
    colors = [
            "#009392",
            "#E9002D",
            "#226E9C",
            ]

    for i, data_model in enumerate(fit_elbos):
        data_elbos = fit_elbos[data_model]
        #(nb_models, nb_rep, nb_epoch)

        nb_iters = data_elbos.shape[2]
        x = [j for j in range(1, nb_iters+1)]

        for j, model_type in enumerate(model_types):
            elbos = data_elbos[j,...]
            axs[i].plot(x, elbos.mean(0), "-", color=colors[j],
                    label="evoVGM_"+model_type.upper(), zorder=j)

            axs[i].fill_between(x,
                elbos.mean(0)-elbos.std(0), 
                elbos.mean(0)+elbos.std(0), 
                color=colors[j],
                alpha=0.2, zorder=j, interpolate=True)
 
        axs[i].set_title(data_model.upper()+"-simulated data")
        axs[i].set_ylim([None, 0])
        axs[i].set_xticks([t for t in range(1, nb_iters+1) \
                if t==1 or t % print_xtick_every==0])
        axs[i].set_xlabel("Iterations")
        axs[i].grid()
        if i==0:
            axs[i].set_ylabel("Evidence Lower Bound")

    plt.legend(bbox_to_anchor=(1.06, 1), 
            loc='upper left', borderaxespad=0.)
    #plt.suptitle(title)
    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)

"""
python summarize_rep_exps.py evaldata5k
"""
if __name__ == '__main__':

    job_code = sys.argv[1]
    verbose = 1

    print("Summarizing {} experiments\n".format(job_code))

    model_types = [
            "jc69",
            "k80", 
            "gtr"
            ]

    data_types = [
            ("jc69", [0.160, 0.160, 0.160, 0.160, 0.160, 0.160], 
                [0.25, 0.25, 0.25, 0.25]), 
            ("k80" , [0.250, 0.125, 0.125, 0.125, 0.125, 0.250],
                [0.25, 0.25, 0.25, 0.25]),
            ("gtr" , [0.160, 0.050, 0.160, 0.090, 0.300, 0.240],
                [0.10, 0.45, 0.30, 0.15])
            ]

    branches = [
            (3, [0.1,0.3,0.45]),
            (4, [0.1,0.3,0.45,0.15]),
            (5, [0.1,0.3,0.45,0.15,0.2])
            ]

    len_alns = [
            100, 
            1000,
            5000,
            ]

    report_n_epochs = 5000
    print_xtick_every = 500

    #report_n_epochs = 1000
    #print_xtick_every = 100

    plot_individuals = True
    Summarize_model_wise = True 
    Summarize_data_wise = True

    output_dir = "../exp_outputs/{}/".format(job_code)

    val_dict = dict()
    fit_dict = dict()

    for data_model, rates, freqs in data_types:
        for nb_seqs, branch_lens in branches:
            for len_aln in len_alns:
                for evomodel in model_types:
 
                    # nb3_l5k_datajc69_evojc69.ini
                    exp_name = "nb{}_l{}_data{}_evo{}".format(
                            nb_seqs, len_aln, data_model, evomodel)

                    if verbose: 
                        print("{}".format(exp_name))

                    output_exp = output_dir+"{}/{}/".format(
                            evomodel, exp_name)

                    results_file = output_exp+\
                            "{}_results_uncompress.pkl".format(
                                    exp_name)
                            #"{}_results.pkl".format(exp_name)

                    # Get data simulation parameters
                    config_file = output_exp+"/{}_conf.ini".format(
                            exp_name)
                    config = configparser.ConfigParser(
                            interpolation=\
                            configparser.ExtendedInterpolation())

                    with open(config_file, "r") as cf:
                        config.read_file(cf)

                    sim_blengths = config.get("data", 
                            "branch_lengths", fallback="0.1,0.1")
                    sim_rates = str_to_values(config.get("data",
                        "rates", fallback="0.16"), 6, cast=float)
                    sim_freqs = str_to_values(config.get("data",
                        "freqs", fallback="0.25"), 4, cast=float)

                    # The order of freqs is different for evoVGM
                    # A G C T
                    sim_freqs_vgm = [sim_freqs[0], sim_freqs[2],
                        sim_freqs[1], sim_freqs[3]]

                    sim_params = dict(
                            b=np.array(str2floats(sim_blengths)),
                            r=np.array(sim_rates),
                            f=np.array(sim_freqs_vgm),
                            k=np.array([[sim_rates[0]/sim_rates[1]]])
                            )

                    if not os.path.isfile(results_file):
                        print("{} doesn't exist!\n".format(
                            results_file))
                    else:
                        #if verbose:
                        #    print("\nLoading scores from file")
                        result_data = load(results_file)
                        rep_results=result_data["rep_results"]
                        # logl of data
                        logl_data=result_data["logl_data"]
                        val_dict[exp_name+"_logl"] = logl_data
                        #print("logl {}".format(logl_data))
 
                        scores = [result["fit_hist"] for\
                                result in rep_results]
                        the_scores = scores

                        # get min number of epoch of all reps 
                        # (maybe some reps stopped before max_iter)
                        # to slice the list of epochs with the same
                        # length 
                        # and be able to cast the list in ndarray        
                        #min_iter = scores[0].shape[1]
                        #for score in scores:
                        #    if min_iter >= score.shape[1]:
                        #        min_iter = score.shape[1]
                        #the_scores = []
                        #for score in scores:
                        #    the_scores.append(score[:,:min_iter])
                        the_scores = np.array(
                                the_scores)[:,:,:report_n_epochs] 
                        #print(the_scores.shape)
                        # (nb_reps, nb_measures, nb_epochs)

                        # get Validation results
                        val_results = [result["val_results"] \
                                for result in rep_results]
                        val_dict[exp_name] = val_results
                        fit_dict[exp_name] = the_scores

                        ## Ploting results
                        ## ###############
                        title = exp_name
                        if plot_individuals:
                            if verbose: 
                                print("Plotting {}..".format(
                                    exp_name))
                            plt_elbo_ll_kl_rep_figure(
                                    the_scores,
                                    output_exp+\
                                            "/{}_fig_exp{}".format(
                                                exp_name,
                                                report_n_epochs),
                                    print_xtick_every=\
                                            print_xtick_every,
                                    usetex=False,
                                    y_limits=[None, 0.],
                                    title=None,
                                    plot_validation=True)

                            estimates = aggregate_estimate_values(
                                    rep_results,
                                    "val_hist_estim",
                                    report_n_epochs)
                            #return a dictionary of arrays

                            plot_fit_estim_dist(
                                    estimates, 
                                    sim_params,
                                    output_exp+\
                                            "/{}_val_estim_dist_exp{}".format(
                                                exp_name, 
                                                report_n_epochs),
                                    y_limits=[-0.1, 1.1],
                                    print_xtick_every=\
                                            print_xtick_every,
                                    usetex=False)

                            plot_fit_estim_corr(
                                    estimates, 
                                    sim_params,
                                    output_exp+\
                                            "/{}_val_estim_corr_exp{}".format(
                                                exp_name, 
                                                report_n_epochs),
                                    y_limits=[-1.1, 1.1],
                                    print_xtick_every=\
                                            print_xtick_every,
                                    usetex=False)
                        #sys.exit()

    # Summarize validation data
    if Summarize_model_wise:
        print("Summarize model wise\n")

        output_sum_model = os.path.join(output_dir, "sum_model_wise")
        makedirs(output_sum_model, mode=0o700, exist_ok=True)

        logl_dict = dict()
        row_index = [nb_seqs for nb_seqs, _ in branches]
        #col_index = pd.MultiIndex.from_product(
        #        [len_alns, ['Real', 'Mean', 'Variance']])
        col_index = pd.MultiIndex.from_product(
                [len_alns, ['Real', 'Mean', 'STD']])

        branch_dict = dict()
        col_index_b = pd.MultiIndex.from_product(
                [len_alns, ['dist', 'corr', 'pval']])

        rates_dict = dict()
        freqs_dict = dict()
        kappa_dict = dict()
        col_index_k = pd.MultiIndex.from_product(
                [len_alns, ['kappa']])

        for data_model, rates, freqs in data_types:
            # Correct the freqs order
            freqs_vgm = [freqs[0], freqs[2], freqs[1], freqs[3]]

            for evomodel in model_types:
                # create the dataframe
                code = "d{}_e{}".format(data_model, evomodel)
                
                df_l = pd.DataFrame("",
                        index=row_index, columns=col_index)
                df_b = pd.DataFrame("", 
                        index=row_index, columns=col_index_b)
                if evomodel == "gtr":
                    df_r = pd.DataFrame("",
                            index=row_index, columns=col_index_b)
                    df_f = pd.DataFrame("", 
                            index=row_index, columns=col_index_b)
                elif evomodel == "k80":
                    df_k = pd.DataFrame("", 
                            index=row_index, columns=col_index_k)
 
                for nb_seqs, branch_lens in branches:
                    for len_aln in len_alns:

                        # nb3_l5k_datajc69_evojc69.ini
                        exp_name = "nb{}_l{}_data{}_evo{}".format(
                                nb_seqs, len_aln, data_model,
                                evomodel)

                        if verbose: 
                            print("{}".format(exp_name))

                        # Log likelihood
                        # ##############
                        df_l[len_aln, "Real"].loc[nb_seqs] =\
                                val_dict[exp_name+"_logl"].item()
                        val_results = val_dict[exp_name]
                        # a list of dictionaries
                        logls = np.array([measures["logl"].item()\
                                for measures in val_results])

                        # val scores of last iteration from fit
                        #fit_results = fit_dict[exp_name]
                        # a list of dictionaries
                        #logls = fit_results[:, 4, -1]
                        #print(logls.shape)

                        df_l[len_aln, "Mean"].loc[nb_seqs] =\
                                logls.mean()
                        #df_l[len_aln, "Variance"].loc[nb_seqs] =\
                        #        logls.var()
                        df_l[len_aln, "STD"].loc[nb_seqs] =\
                                logls.std()

                        # Branches
                        # ########
                        bls = np.array([measures["b"].numpy()\
                                for measures in val_results])
                        bls = bls.mean(0).flatten()
                        df_b[len_aln, "dist"].loc[nb_seqs] = pdist(
                                np.vstack((branch_lens, bls)),
                                'euclidean')[0]
                        corr_b = pearsonr(branch_lens, bls)
                        df_b[len_aln,"corr"].loc[nb_seqs] = corr_b[0]
                        df_b[len_aln,"pval"].loc[nb_seqs] = corr_b[1]

                        # Rates & freqs
                        # #############
                        if evomodel == "gtr":
                            rs = np.array([measures["r"].numpy()\
                                    for measures in val_results])
                            rs = rs.mean(0).flatten()

                            fs = np.array([measures["f"].numpy()\
                                    for measures in val_results])
                            fs = fs.mean(0).flatten()

                            df_r[len_aln, "dist"].loc[nb_seqs] =\
                                    pdist(np.vstack((rates, rs)), 
                                            'euclidean')[0]

                            df_f[len_aln, "dist"].loc[nb_seqs] =\
                                    pdist(np.vstack((freqs_vgm,
                                        fs)), 'euclidean')[0]

                            if data_model != "jc69":
                                corr_r = pearsonr(rates, rs)
                                df_r[len_aln, "corr"].loc[nb_seqs] =\
                                        corr_r[0]
                                df_r[len_aln, "pval"].loc[nb_seqs] =\
                                        corr_r[1]

                                corr_f = pearsonr(freqs_vgm, fs)
                                df_f[len_aln, "corr"].loc[nb_seqs] =\
                                        corr_f[0]
                                df_f[len_aln, "pval"].loc[nb_seqs] =\
                                        corr_f[1]

                        elif evomodel == "k80":
                            ks = np.array([measures["k"].numpy()\
                                    for measures in val_results])
                            ks = ks.mean(0).flatten()[0]
 
                            df_k[len_aln, "kappa"].loc[nb_seqs] = ks

                #print(code)

                logl_dict[code] = df_l
                #print(df_l)
                branch_dict[code] = df_b
                #print(df_b)
                if evomodel == "gtr":
                    rates_dict[code] = df_r
                    #print(df_r)
                    freqs_dict[code] = df_f
                    #print(df_f)
                elif evomodel == "k80":
                    kappa_dict[code] = df_k
                    #print(df_k)
                #print()
 
        #write_dict_df(logl_dict,
        #output_sum_model+"/model_wise_logl_var.txt")
        write_dict_df(logl_dict,
                output_sum_model+"/model_wise_logl_std.txt")
        write_dict_df(branch_dict,
                output_sum_model+"/model_wise_branches.txt")
        write_dict_df(rates_dict,
                output_sum_model+"/model_wise_rates.txt")
        write_dict_df(freqs_dict,
                output_sum_model+"/model_wise_freqs.txt")
        write_dict_df(kappa_dict,
                output_sum_model+"/model_wise_kappa.txt")
 
    if Summarize_data_wise:
        print("Summarize data wise\n")
        # outputdir 
        output_sum_data = os.path.join(output_dir, "sum_data_wise")
        makedirs(output_sum_data, mode=0o700, exist_ok=True)

        evo_data = dict()
        row_index = ['Real'] + model_types
        #col_index = pd.MultiIndex.from_product(
        #        [[d for d,_,_ in data_types], ['Mean', 'Variance']])
        col_index = pd.MultiIndex.from_product(
                [[d for d,_,_ in data_types], ['Mean', 'STD']])

        for nb_seqs, branch_lens in branches:
            for len_aln in len_alns:
                # create the dataframe
                code = "nb{}_l{}".format(nb_seqs, len_aln)
                df_l = pd.DataFrame("",
                        index=row_index,
                        columns=col_index)

                fit_elbos = dict()

                for data_model, _, _ in data_types:
                    data_elbos = []
                    for evomodel in model_types:

                        # nb3_l5k_datajc69_evojc69.ini
                        exp_name = "nb{}_l{}_data{}_evo{}".format(
                                nb_seqs, len_aln, data_model,
                                evomodel)

                        if verbose: 
                            print("{}".format(exp_name))

                        df_l[data_model, "Mean"].loc["Real"] =\
                                val_dict[exp_name+"_logl"].item()

                        val_results = val_dict[exp_name]
                        # a list of dictionaries
                        logls = np.array([measures["logl"].item() \
                                for measures in val_results])

                        # val scores of last iteration from fit
                        #fit_results = fit_dict[exp_name]
                        # a list of dictionaries
                        #logls = fit_results[:, 4, -1]
                        #print(logls.shape)

                        df_l[data_model, "Mean"].loc[evomodel] =\
                                logls.mean()
                        #df_l[data_model, "Variance"].loc[evomodel]=\
                        #        logls.var()
                        df_l[data_model, "STD"].loc[evomodel] =\
                                logls.std()

                        # Get fit elbo
                        fit_results = fit_dict[exp_name]
                        # a list of dictionaries
                        elbos = fit_results[:, 0, :]
                        data_elbos.append(elbos)

                    fit_elbos[data_model] = np.array(data_elbos)
                    # print(fit_elbos[data_model].shape)
                    # (nb_models, nb_rep, nb_epoch)

                print(code)
                if True:
                    out_file = output_sum_data+"/{}_itr{}".format(
                            code, report_n_epochs) 
                    plot_elbos_data_wise(
                            fit_elbos,
                            model_types,
                            out_file,
                            print_xtick_every=print_xtick_every,
                            title=code)

                evo_data[code] = df_l

        #write_dict_df(evo_data,
        #output_sum_data+"/data_wise_logl_var.txt")
        write_dict_df(evo_data,
                output_sum_data+"/data_wise_logl_std.txt")
