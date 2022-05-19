#!/usr/bin/env python

from evoVGM.reports import plt_elbo_ll_kl_rep_figure

import sys
import os
import os.path
from os import makedirs

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist

from joblib import dump, load

import matplotlib.pyplot as plt

__author__ = "amine"


def plot_hyperparams(
        hyper_scores,
        hyper_values,
        out_file,
        usetex=False,
        print_xtick_every=20,
        title="title",
        plot_validation=False):

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    sizefont = 10

    nb_evals = hyper_scores.shape[0]

    f, axs = plt.subplots(1, nb_evals, figsize=(7*nb_evals, 5))

    plt.rcParams.update({'font.size':sizefont, 'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.07, hspace=0.1)

    # Add more colors if nb models > 3 or use a template
    elbo_color = "#E9002D" #sharop red
    ll_color =   "#226E9C"  # darker blue
    kl_color =   "#7C1D69"  # pink

    elbo_color_v = "tomato"
    ll_color_v =   "#009ADE" # light blue
    kl_color_v =   "#AF58BA"  # light pink

    nb_iters = hyper_scores.shape[-1] 
    x = [j for j in range(1, nb_iters+1)]
 
    for i, hyper_value in enumerate(hyper_values):
        scores = hyper_scores[i,...]

        ax2 = axs[i].twinx()
        # plot means
        axs[i].plot(x, scores[:,0,:].mean(0), "-", color=elbo_color, 
                label="ELBO", zorder=6) # ELBO train
        axs[i].plot(x, scores[:,1,:].mean(0), "-", color=ll_color,
                label="LogL", zorder=4) # LL train
        ax2.plot(x, scores[:,2,:].mean(0), "-", color=kl_color,
                label="KL_qp", zorder=4) # KL train

        # plot stds 
        axs[i].fill_between(x,
                scores[:,0,:].mean(0)-scores[:,0,:].std(0), 
                scores[:,0,:].mean(0)+scores[:,0,:].std(0), 
                color=elbo_color,
                alpha=0.2, zorder=5, interpolate=True)

        axs[i].fill_between(x,
                scores[:,1,:].mean(0)-scores[:,1,:].std(0), 
                scores[:,1,:].mean(0)+scores[:,1,:].std(0),
                color=ll_color,
                alpha=0.2, zorder=3, interpolate=True)

        ax2.fill_between(x,
                scores[:,2,:].mean(0)-scores[:,2,:].std(0), 
                scores[:,2,:].mean(0)+scores[:,2,:].std(0), 
                color=kl_color,
                alpha=0.2, zorder=-6, interpolate=True)

        # plot validation
        if plot_validation:
            axs[i].plot(x, scores[:,3,:].mean(0), "-.",
                    color=elbo_color_v,
                    label="ELBO_val", zorder=2) # ELBO val
            axs[i].plot(x, scores[:,4,:].mean(0), "-.",
                    color=ll_color_v,
                    label="LogL_val", zorder=0) # LL val
            ax2.plot(x, scores[:,5,:].mean(0), "-.", 
                    color=kl_color_v,
                    label="KL_qp_val", zorder=2) # KL val
            
            axs[i].fill_between(x,
                    scores[:,3,:].mean(0)-scores[:,3,:].std(0), 
                    scores[:,3,:].mean(0)+scores[:,3,:].std(0), 
                    color=elbo_color_v,
                    alpha=0.1, zorder=1, interpolate=True)

            axs[i].fill_between(x,
                    scores[:,4,:].mean(0)-scores[:,4,:].std(0), 
                    scores[:,4,:].mean(0)+scores[:,4,:].std(0), 
                    color=ll_color_v,
                    alpha=0.1, zorder=2, interpolate=True)

            ax2.fill_between(x,
                    scores[:,5,:].mean(0)-scores[:,5,:].std(0), 
                    scores[:,5,:].mean(0)+scores[:,5,:].std(0), 
                    color= kl_color_v,
                    alpha=0.1, zorder=1, interpolate=True)

        axs[i].set_zorder(ax2.get_zorder()+1)
        axs[i].set_frame_on(False)

        axs[i].set_title(hyper_value)
        #axs[i].set_ylim(y_limits)
        #axs[i].set_ylim([None, 0])
        #axs[i].set_ylim([-10000, 0])
        #print( np.min(hyper_scores[:,:,0,:].flatten()) )
        axs[i].set_ylim([ np.min(hyper_scores[:,:,0,:].flatten()), 0])
        axs[i].set_xticks([t for t in range(1, nb_iters+1) if t==1 or\
                t % print_xtick_every==0])
        axs[i].set_xlabel("Iterations")
        axs[i].grid()

        ax2.set_ylim([ 
            np.min(hyper_scores[:,:,2,:].flatten()),
            np.max(hyper_scores[:,:,2,:].flatten())])

        if i != 0:
            axs[i].set(yticklabels=[])
        if i != nb_evals - 1:
            ax2.set(yticklabels=[])

    handles,labels = [],[]
    for ax in f.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    plt.legend(handles, labels, bbox_to_anchor=(1.15, 1), 
            loc='upper left', borderaxespad=0.)
    #plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)


if __name__ == '__main__':

    job_code = sys.argv[1]
    verbose = 1

    print("Summarizing {} experiments\n".format(job_code))

    max_iter = "5000"
    n_reps = "10"

    hyper_types = {
            "kl": ("alpha_kl", (0.0001, 0.001, 0.01, 0.1), 0.0001),
            "hs": ("hidden_size", (4, 16, 32, 64), 32),
            "ns": ("nb_samples", (1, 10, 100, 1000), 100),
            "lr": ("learning_rate",
                (0.00005, 0.0005, 0.005, 0.05), 0.005),
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

    report_n_epochs = 200

    # Choose here type of data and model
    ind_model = 1
    evomodel = model_types[ind_model]
    data_model, rates, freqs = data_types[ind_model]
    #
    nb_seqs, branch_lens = branches[1]
    len_aln = len_alns[1]

    output_dir = "../exp_outputs/{}/".format(job_code)

    output_sum = os.path.join(output_dir, "summarize")
    makedirs(output_sum, mode=0o700, exist_ok=True)

    #val_dict = dict()
    fit_dict = dict()

    for hyper_code in hyper_types:
        hyper_name = hyper_types[hyper_code][0]
        hyper_values = hyper_types[hyper_code][1]

        hyper_scores = []
        evo_hyper = "{}_{}".format(evomodel, hyper_code) 
        for hyper_value in hyper_values:
            # evogtr_hs_4 (old names)
            #exp_name = "evo{}_{}_{}".format(evomodel,
            #        hyper_code, hyper_value)
 
            # nb3_l5k_datajc69_evojc69_hs4.ini
            exp_name = "nb{}_l{}_data{}_evo{}_{}{}".format(
                    nb_seqs, len_aln, data_model, evomodel,
                    hyper_code, hyper_value)

            print(exp_name)

            output_exp = output_dir+"{}/{}/".format(
                    evomodel, exp_name)

            results_file = output_exp+"{}_results.pkl".format(
                    exp_name)

            if not os.path.isfile(results_file):
                print("{} doesn't exist!\n".format(results_file))
            else:
                #if verbose: print("\nLoading scores from file")
                result_data = load(results_file)
                rep_results=result_data["rep_results"]
                # logl of data
                logl_data=result_data["logl_data"]
                #val_dict[exp_name+"_logl"] = logl_data
                #print("logl {}".format(logl_data))

                scores = [result["fit_hist"] for\
                        result in rep_results]
                the_scores = scores

                # get min number of epoch of all reps 
                # (maybe some reps stopped before max_iter)
                # to slice the list of epochs with the same length 
                # and be able to cast the list in ndarray        
                #min_iter = scores[0].shape[1]
                #for score in scores:
                #    if min_iter >= score.shape[1]: min_iter = score.shape[1]
                #the_scores = []
                #for score in scores:
                #    the_scores.append(score[:,:min_iter])
                the_scores=np.array(the_scores)[:,:,:report_n_epochs] 
                #print(the_scores.shape) # (nb_reps, nb_measures, nb_epochs)

                # get Validation results
                #val_results = [result["val_results"] for\
                #        result in rep_results]
                #val_dict[exp_name] = val_results

                hyper_scores.append(the_scores)

                ## Ploting results
                ## ###############
                title = exp_name
                if False:
                    if verbose: 
                        print("Plotting {}..".format(exp_name))
                    plt_elbo_ll_kl_rep_figure(
                            the_scores,
                            output_exp+"/{}_exp{}_fig".format(
                                exp_name, report_n_epochs),
                            print_xtick_every=200,
                            usetex=False,
                            y_limits=[-100, 0],
                            title=title,
                            plot_validation=True)
                #sys.exit()
        hyper_scores = np.array(hyper_scores)
        # print(hyper_scores.shape)
        # (nb_hyper_values, nb_reps, nb_measures, epochs)

        if True:
            if verbose: 
                print("Plotting {}..".format(evo_hyper))
            out_file = output_sum+"/{}".format(evo_hyper) 
            plot_hyperparams(
                    hyper_scores,
                    hyper_values,
                    out_file,
                    print_xtick_every=20,
                    title=evo_hyper,
                    plot_validation=True)
