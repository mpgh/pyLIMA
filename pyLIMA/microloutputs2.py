# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:38:14 2015

@author: ebachelet
"""
from __future__ import division
from datetime import datetime
from collections import OrderedDict
import collections
import copy
import json
import cycler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap

import numpy as np
from astropy.time import Time
from scipy.stats.distributions import t as student
import os

from pyLIMA import microltoolbox
from pyLIMA import microlstats
from pyLIMA import microlcaustics
from pyLIMA import detection_zone_area
from pyLIMA import telescopes
from pyLIMA import event
import pyprind

plot_lightcurve_windows = 0.2
plot_residuals_windows = 0.21
MAX_PLOT_TICKS = 2
MARKER_SYMBOLS = np.array([['o', '.', '*', 'v', '^', '<', '>', 's', 'p', 'd', 'x'] * 10])


# plt.style.use('ggplot')


def latex_output(fit, output_directory):
    """Function to output a LaTeX format table of the fit parameters"""

    event_name = fit.event.name
    fit_type = fit.method

    file_path = os.path.join(output_directory, event_name + '_' + fit_type + '_results.tex')

    t = open(file_path, 'w')

    t.write('\\begin{table}[h!]\n')
    t.write('\\centering\n')
    t.write(
        '\\caption{Best model parameters} \label{tab:fitparams}\n')
    t.write('\\begin{tabular}{lll}\n')
    t.write('\\hline\n')
    t.write('\\hline\n')

    if fit_type == 'MCMC':
        t.write('Parameters&Value(best model)&Errors([16,50,84] range)')
        t.write('\\hline\n')

        mcmc_chains = fit.MCMC_chains
        best_model_index = np.argmax(mcmc_chains[:, -1])


        for index, key in enumerate(fit.model.model_dictionnary):
            best_param = mcmc_chains[best_model_index, index]
            percent_34 = np.percentile(mcmc_chains[:, index], 16)
            percent_50 = np.percentile(mcmc_chains[:, index], 50)
            percent_84 = np.percentile(mcmc_chains[:, index], 84)

            t.write(
                    key + '&' + str(best_param) + '&[' + str(percent_34) + ',' + str(percent_50) + ',' + str(
                    percent_84) + ']\\\\\n')

        t.write('Chi2&' + str(-2*mcmc_chains[best_model_index, -1]) + '&0\\\\\n')

    else:
        t.write('Parameters&Value&Errors')
        t.write('\\hline\n')

        for index, key in enumerate(fit.model.model_dictionnary):
            t.write(key + '&' + str(fit.fit_results[index]) + '&' + str(
                    fit.fit_covariance.diagonal()[index] ** 0.5) + '\\\\\n')

        t.write('Chi2&' + str(fit.fit_results[-1]) + '&0\\\\\n')


    t.write('\\hline\n')
    t.write('\\end{tabular}\n')
    t.write('\\end{table}\n')

    t.close()


def pdf_output(fit, output_directory):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output_directory + fit.event.name + '.pdf') as pdf:
        figure_1 = fit.outputs.figure_lightcurve
        pdf.savefig(figure_1)

        figure_2 = fit.outputs.figure_geometry
        pdf.savefig(figure_2)

        if 'figure_distributions' in fit.outputs._fields:
            figure_3 = fit.outputs.figure_distributions
            pdf.savefig(figure_3)
        pdf_details = pdf.infodict()
        pdf_details['Title'] = fit.event.name + '_pyLIMA'
        pdf_details['Author'] = 'Produced by pyLIMA'
        pdf_details['Subject'] = 'A microlensing fit'

        pdf_details['CreationDate'] = datetime.today()


def statistical_outputs(fit):
    """Compute statistics to estimate the fit quality

        :param object fit: a fit object. See the microlfits for more details.

        :return: a namedtuple containing the following attributes :

                  fit_parameters : an namedtuple object containing all the fitted parameters

                  fit_errors : an namedtuple object containing all the fitted parameters errors

                  fit_correlation_matrix : a numpy array representing the fitted parameters
                  correlation matrix

                  figure_lightcurve : a two matplotlib figure showing the data and model and the
                  correspoding residuals

        :rtype: object
    """
    fig_size = [15, 5]
    figure_stats = plt.figure(figsize=(fig_size[0], fig_size[1]))

    best_parameters = fit.fit_results
    best_model_pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(best_parameters)

    telescope_residuals = fit.all_telescope_residuals(best_model_pyLIMA_parameters)

    telescope_Kolmogorv_Smirnov_residuals_test = []
    telescope_Anderson_Darling_residuals_test = []
    telescope_Shapiro_Wilk_residuals_test = []

    telescope_chi2 = []
    telescope_chi2_sur_dof = []
    telescope_BIC = []
    telescope_AIC = []
    count = 0
    for telescope in fit.event.telescopes:
        residuals = telescope_residuals[count]

        telescope_residuals.append(residuals)

        Kolmogorov_Smirnov = microlstats.normal_Kolmogorov_Smirnov(residuals)

        Anderson_Darling = microlstats.normal_Anderson_Darling(residuals)

        Shapiro_Wilk = microlstats.normal_Shapiro_Wilk(residuals)

        telescope_Kolmogorv_Smirnov_residuals_test.append(Kolmogorov_Smirnov)
        telescope_Anderson_Darling_residuals_test.append(Anderson_Darling)
        telescope_Shapiro_Wilk_residuals_test.append(Shapiro_Wilk)

        chi2_sur_dof = microlstats.normalized_chi2((residuals ** 2).sum(), len(residuals),
                                                   len(fit.model.parameters_boundaries) + 2)
        BIC = 0.0
        AIC = 0.0

        telescope_chi2.append((residuals ** 2).sum())
        telescope_chi2_sur_dof.append(chi2_sur_dof)
        telescope_BIC.append(BIC)
        telescope_AIC.append(AIC)
        count += 1
    total_residuals = fit.residuals_LM(best_parameters)

    Kolmogorov_Smirnov = microlstats.normal_Kolmogorov_Smirnov(total_residuals)

    Anderson_Darling = microlstats.normal_Anderson_Darling(total_residuals)

    Shapiro_Wilk = microlstats.normal_Shapiro_Wilk(total_residuals)

    telescope_Kolmogorv_Smirnov_residuals_test.append(Kolmogorov_Smirnov)
    telescope_Anderson_Darling_residuals_test.append(Anderson_Darling)
    telescope_Shapiro_Wilk_residuals_test.append(Shapiro_Wilk)

    chi2_sur_dof = microlstats.normalized_chi2(best_parameters[-1], len(total_residuals),
                                               len(fit.model.parameters_boundaries) + 2)
    BIC = microlstats.Bayesian_Information_Criterion(best_parameters[-1], len(total_residuals),
                                                     len(fit.model.parameters_boundaries) + 2)
    AIC = microlstats.Akaike_Information_Criterion(best_parameters[-1], len(fit.model.parameters_boundaries) + 2)

    telescope_chi2.append(best_parameters[-1])
    telescope_chi2_sur_dof.append(chi2_sur_dof)
    telescope_BIC.append(BIC)
    telescope_AIC.append(AIC)

    raw_labels = [i.name for i in fit.event.telescopes]
    raw_labels += ['All site']
    column_labels = ['Kolmogorov-Smirnov\n(KS_stat,p_value)', 'Anderson-Darling\n(AD_stat,p value)',
                     'Shapiro-Wilk\n(SW_stat,p_value)', 'chi2', 'chi2_dof', 'BIC', 'AIC']
    table_val = []
    table_colors = []
    colors_dictionary = {0: 'r', 1: 'y', 2: 'g'}

    for i in range(len(raw_labels)):
        table_val.append([np.round(telescope_Kolmogorv_Smirnov_residuals_test[i][:2], 3),
                          np.round(telescope_Anderson_Darling_residuals_test[i][:2], 3),
                          np.round(telescope_Shapiro_Wilk_residuals_test[i][:2], 3),
                          np.round(telescope_chi2[i], 3), np.round(telescope_chi2_sur_dof[i][0], 3),
                          np.round(telescope_BIC[i], 3), np.round(telescope_AIC[i], 3)])

        table_colors.append([colors_dictionary[telescope_Kolmogorv_Smirnov_residuals_test[i][2]],
                             colors_dictionary[telescope_Anderson_Darling_residuals_test[i][2]],
                             colors_dictionary[telescope_Shapiro_Wilk_residuals_test[i][2]],
                             'w',
                             colors_dictionary[telescope_chi2_sur_dof[i][1]],
                             'w',
                             'w',
                             ])

    # table_val = np.round(table_val, 5).tolist()



    table_axes = figure_stats.add_subplot(111, frameon=False)

    the_table = table_axes.table(cellText=table_val,
                                 rowLabels=raw_labels, cellColours=table_colors,
                                 colLabels=column_labels, loc='center left')
    table_axes.get_yaxis().set_visible(False)
    table_axes.get_xaxis().set_visible(False)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(fig_size[0] * 3 / 4.0 / np.log10(len(fit.model.model_dictionnary.keys())))
    the_table.scale(0.75, 0.75)
    title = fit.model.event.name + ' : ' + fit.model.model_type
    figure_stats.suptitle(title, fontsize=30 * fig_size[0] / len(title))


def fit_outputs(fit):
    """Standard outputs.

    :param object fit: a fit object. See the microlfits for more details.

    :return: a namedtuple containing the following attributes :

              fit_parameters : an namedtuple object containing all the fitted parameters

              fit_errors : an namedtuple object containing all the fitted parameters errors

              fit_correlation_matrix : a numpy array representing the fitted parameters
              correlation matrix

              figure_lightcurve : a two matplotlib figure showing the data and model and the
              correspoding residuals

    :rtype: object
    """
    # Change matplotlib default colors
    n = len(fit.event.telescopes)
    color = plt.cm.jet(np.linspace(0.01, 0.99, n))  # This returns RGBA; convert:
    # hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
    #                tuple(color[:, 0:-1]))
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]

    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)
    # hexcolor[0] = '#000000'




    if (fit.method == 'LM') or (fit.method=='TRF'):
        # prepare a list of fake telescopes
        create_the_fake_telescopes(fit, fit.fit_results)

        results = parameters_result(fit)

        figure_trajectory = plot_geometry(fit)
        figure_table = parameters_table(fit)
        covariance_matrix = fit.fit_covariance
        errors = fit_errors(fit)
        figure_lightcurve = LM_plot_lightcurves(fit)
        if 'PL' in fit.model.model_type:
            detection_zone = plot_detection_zone_area(fit)

            key_outputs = ['fit_parameters', 'fit_errors', 'fit_correlation_matrix', 'figure_lightcurve', 'figure_geometry', 'detection_zone', 'figure_table']
            values_outputs = [results, errors, covariance_matrix, figure_lightcurve, figure_trajectory, detection_zone, figure_table]
        else:
            key_outputs = ['fit_parameters', 'fit_errors', 'fit_correlation_matrix', 'figure_lightcurve', 'figure_geometry', 'figure_table']
            values_outputs = [results, errors, covariance_matrix, figure_lightcurve, figure_trajectory, figure_table]
    if fit.method == 'DE':
        # prepare a list of fake telescopes
        create_the_fake_telescopes(fit, fit.fit_results)

        results = parameters_result(fit)
        figure_trajectory = plot_geometry(fit)
        figure_table = parameters_table(fit)
        covariance_matrix = fit.fit_covariance
        errors = fit_errors(fit)

        figure_lightcurve = LM_plot_lightcurves(fit)

        figure_distributions = plot_distributions(fit, fit.DE_population)
        key_outputs = ['fit_parameters', 'fit_errors', 'fit_correlation_matrix', 'figure_lightcurve', 'figure_geometry',
                       'figure_table','figure_distributions']
        values_outputs = [results, errors, covariance_matrix, figure_lightcurve, figure_trajectory, figure_table,
                          figure_distributions]

    if fit.method == 'MCMC':

        mcmc_chains = fit.MCMC_chains

        best_chain = copy.copy(mcmc_chains[np.argmax(mcmc_chains[:, -1])])
        best_chain[-1] *= -2 #likelihood to chi2
        # prepare a list of fake telescopes
        create_the_fake_telescopes(fit, best_chain)


        results = parameters_result(fit,best_chain)
        covariance_matrix = MCMC_covariance(mcmc_chains)
        errors = fit_errors(fit,covariance_matrix)

        index = np.random.choice(range(len(mcmc_chains)), 12)
        index = np.r_[index, np.argmax(mcmc_chains[:, -1])]

        best_chains = mcmc_chains[index]
        best_chains = best_chains[best_chains[:, -1].argsort(),]


        figure_lightcurve = MCMC_plot_lightcurves(fit, best_chains)
        figure_trajectory = plot_geometry(fit)
        figure_table = parameters_table(fit)

        figure_distributions = plot_distributions(fit, mcmc_chains)

        key_outputs = ['fit_parameters', 'fit_errors', 'fit_correlation_matrix', 'figure_lightcurve', 'figure_geometry',
                       'figure_table','figure_distributions']
        values_outputs = [results, errors, covariance_matrix, figure_lightcurve, figure_trajectory, figure_table,
                           figure_distributions]

    outputs = collections.namedtuple('Fit_outputs', key_outputs)


    count = 0
    for key in key_outputs:
        setattr(outputs, key, values_outputs[count])
        count += 1

    return outputs


def complete_MCMC_parameters(fit, parameters):

    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(parameters)
    chichi = parameters[-1]
    parameters = parameters[:-1].tolist()
    for index,telescope in enumerate(fit.event.telescopes):

        _, fs, fb = fit.model.compute_the_microlensing_model(telescope, pyLIMA_parameters)
        parameters.append(fs)
        parameters.append(fb)

    return parameters+[chichi]

def create_the_fake_telescopes(fit,parameters):


    fit.event.fake_telescopes = []
    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(parameters)

    telescopes_ground = np.array([[i,fit.event.telescopes[i].n_data()] for i
                                  in range(len(fit.event.telescopes)) if fit.event.telescopes[i].location == 'Earth'])

    telescopes_index = [telescopes_ground[np.argmax(telescopes_ground[:,1]),0]]
    telescopes_space = [i for i in range(len(fit.event.telescopes)) if fit.event.telescopes[i].location == 'Space']

    telescopes_index += telescopes_space

    if 0 not in telescopes_index:
        telescopes_index = np.r_[0, telescopes_index].astype(int)
    telescopes_index =  np.sort(telescopes_index)

    for telescope_index in telescopes_index:

        reference_telescope = copy.copy(fit.event.telescopes[telescope_index])
        telescope_time = fit.event.telescopes[telescope_index].lightcurve_magnitude[:, 0]

        if fit.event.telescopes[telescope_index].location == 'Space':

            time = np.arange(np.max([np.min(telescope_time), pyLIMA_parameters.to - 3 * pyLIMA_parameters.tE]),
                             np.min([np.max(telescope_time), pyLIMA_parameters.to + 3 * pyLIMA_parameters.tE]), 0.01)
        else:
            time = np.arange(np.min([np.min(telescope_time), pyLIMA_parameters.to - 3 * pyLIMA_parameters.tE]),
                             np.max([np.max(telescope_time), pyLIMA_parameters.to + 3 * pyLIMA_parameters.tE]), 0.01)


        reference_telescope.lightcurve_magnitude = np.array([time, [0] * len(time), [0] * len(time)]).T
        reference_telescope.lightcurve_flux = reference_telescope.lightcurve_in_flux()

        if fit.model.parallax_model[0] != 'None':
            reference_telescope.compute_parallax(fit.event, fit.model.parallax_model)

        fit.event.fake_telescopes.append(reference_telescope)

def plot_distributions(fit, mcmc_chains):
    """ Plot the fit parameters distributions.
        Only plot the best mcmc_chains are plotted.
        :param fit: a fit object. See the microlfits for more details.
        :param mcmc_best: a numpy array representing the best (<= 6 sigma) mcmc chains.
        :return: a multiple matplotlib subplot representing the parameters distributions (2D slice +
        histogram)
        :rtype: matplotlib_figure
    """

    fig_size = [10, 10]

    max_plot_ticks = MAX_PLOT_TICKS
    dimensions = len(mcmc_chains[0])-1

    figure_distributions, axes = plt.subplots(dimensions, dimensions, figsize=(fig_size[0], fig_size[1]))
    keys = list(fit.model.model_dictionnary)

    chains = np.copy(mcmc_chains)
    chains[:,0] -= 2450000
    for i in range(len(chains[0]) - 1):

        chain_i = chains[:, i]
        if i == 0:
            #chain_i -= 2450000
            pass
        for j in range(len(chains[0]) - 1):

            chain_j = chains[:, j]

            if j == 0:
                #chain_j -= 2450000
                pass
            if j == i:

                axes[j, i].hist(chain_i, 50, histtype='step')

            else:

                axes[j, i].hist2d(chain_i, chain_j, 50, norm=LogNorm())


            axes[j, i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            axes[j, i].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            axes[j, i].xaxis.set_major_locator(MaxNLocator(2))
            axes[j, i].yaxis.set_major_locator(MaxNLocator(2))
            #axes[j, i].set_xticks([np.percentile(chain_i, 1), np.percentile(chain_i, 99)])
            #axes[j, i].set_yticks([np.percentile(chain_j, 99), np.percentile(chain_j, 99)])

           



            if j > i:
                figure_distributions.delaxes(axes[i, j])

            if (i == 0):

                if j != len(chains[0]) - 2:
                    axes[j, i].set_xticks([])

                axes[j, i].set_ylabel(keys[j])

            if j == len(chains[0]) - 2:

                if i != 0:
                    axes[j, i].set_yticks([])
                axes[j, i].set_xlabel(keys[i])

            if (i != 0) and (j != len(chains[0]) - 2):
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([])

            if (i == 0) and (j == 0):
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([])
                axes[j, i].set_ylabel(None)

    figure_distributions.subplots_adjust(hspace=0.1, wspace=0.1)

    return figure_distributions



def parameters_result(fit, parameters = None):
    """ Produce a namedtuple object containing the fitted parameters in the fit.fit_results.

    :param fit: a fit object. See the microlfits for more details.
    :param fit_parameters: a namedtuple object containing the fitted parameters.
    :rtype: object
    """

    fit_parameters = collections.namedtuple('Parameters', fit.model.model_dictionnary.keys())

    if parameters is not None :

        pass

    else:

        parameters = fit.fit_results

    for parameter in fit.model.model_dictionnary.keys():

        try:
            setattr(fit_parameters, parameter, parameters[fit.model.model_dictionnary[parameter]])
        except:
            pass

    setattr(fit_parameters, 'chichi', parameters[-1])
    return fit_parameters


def MCMC_covariance(mcmc_chains):
    """ Estimate the covariance matrix from the mcmc_chains

    :param mcmc_chains: a numpy array representing the mcmc chains.
    :return : a numpy array representing the covariance matrix of your MCMC sampling.
    :rtype: array_like
    """
    esperances = []
    for i in range(mcmc_chains.shape[1] - 1):
        esperances.append(mcmc_chains[:, i] - np.median(mcmc_chains[:, i]))

    covariance_matrix = np.zeros((mcmc_chains.shape[1] - 1, mcmc_chains.shape[1] - 1))

    for i in range(mcmc_chains.shape[1] - 1):
        for j in np.arange(i, mcmc_chains.shape[1] - 1):
            covariance_matrix[i, j] = 1 / (len(mcmc_chains) - 1) * np.sum(
                esperances[i] * esperances[j])
            covariance_matrix[j, i] = 1 / (len(mcmc_chains) - 1) * np.sum(
                esperances[i] * esperances[j])

    return covariance_matrix


def fit_errors(fit,covariance_matrix=None):
    """ Estimate the parameters errors from the covariance matrix.

    :param fit: a fit object. See the microlfits for more details.
    :return: a namedtuple object containing the square roots of parameters variance.
    :rtype: object
    """

    if covariance_matrix is not None:

        pass

    else:

        covariance_matrix = fit.fit_covariance


    keys = ['err_' + parameter for parameter in fit.model.model_dictionnary.keys()]
    parameters_errors = collections.namedtuple('Errors_Parameters', keys)
    errors = covariance_matrix.diagonal() ** 0.5

    for i in fit.model.model_dictionnary.keys():
        try:
            setattr(parameters_errors, 'err_' + i, errors[fit.model.model_dictionnary[i]])
        except:
            pass
    return parameters_errors


def cov2corr(covariance_matrix):
    """Covariance matrix to correlation matrix.

    :param array_like covariance_matrix: a (square) numpy array representing the covariance matrix

    :return: a (square) numpy array representing the correlation matrix
    :rtype: array_like

    """

    covariance_diagonal = np.sqrt(covariance_matrix.diagonal())
    correlation_matrix = ((covariance_matrix.T / covariance_diagonal).T) / covariance_diagonal

    return correlation_matrix


def MCMC_plot_lightcurves(fit, mcmc_best):
    """Plot 35 models from the mcmc_best sample. This is made to have  35 models equally spaced
    in term of objective funtion (~chichi)

    :param fit: a fit object. See the microlfits for more details.
    :param mcmc_best: a numpy array representing the best (<= 6 sigma) mcmc chains.
    :return: a two matplotlib subplot showing the data and 35 models and the residuals
    corresponding to the best model.
    :rtype: matplotlib_figure
    """

    figure_lightcurves, figure_axes = initialize_plot_lightcurve(fit)


    telescope_index = 0
    telescope_reference_name = fit.event.telescopes[telescope_index].name

    for telescope_index, telescope in enumerate(fit.event.fake_telescopes):

        telescope_index_color = [i for i in range(len(fit.event.telescopes)) if
                                 fit.event.telescopes[i].name == telescope.name][0]

        for idx,parameters in enumerate(mcmc_best):

            if len(parameters[:-1])<len(fit.model.model_dictionnary):
                parameters = complete_MCMC_parameters(fit, parameters)

            pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(parameters)
            parameters_to_plot = copy.copy(parameters)

            # put all flux parameters to the telescope 0 scale

            for index, telescope in enumerate(fit.event.telescopes):

                telescope_name = telescope.name
                parameters_to_change = [(i, name) for i, name in enumerate(pyLIMA_parameters._fields) if
                                        telescope_name in name]

                for parameter in parameters_to_change:
                    parameters_to_plot[parameter[0]] = getattr(pyLIMA_parameters, parameter[1].replace(telescope_name,
                                                                                                       telescope_reference_name))


            plot_model(figure_axes[0], fit,
                       parameters=parameters_to_plot, telescope_index=telescope_index, model_color='grey',
                       model_alpha=0.25, label=False)

        if telescope_index == 0:

            plot_model(figure_axes[0], fit,
               parameters=parameters_to_plot, telescope_index=telescope_index,
               model_color=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index_color])
        else:

            plot_model(figure_axes[0], fit,
                       parameters=parameters_to_plot, telescope_index=telescope_index,
                       model_color=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index_color],label=False)


    plot_align_data(figure_axes[0], fit, telescope_index=0, parameters=parameters)

    plot_residuals(figure_axes[1], fit, parameters=parameters)

    figure_axes[0].legend(numpoints=1, loc='best',
                          fancybox=True, framealpha=0.5)
    return figure_lightcurves


def LM_plot_lightcurves(fit):

    """Plot the aligned datasets and the best fit on the first subplot figure_axes[0] and residuals
    on the second subplot figure_axes[1].

    :param object fit: a fit object. See the microlfits for more details.
    :return: a figure representing data+model and residuals.
    :rtype: matplotlib_figure

    """

    figure, figure_axes = initialize_plot_lightcurve(fit)

    try:
        best_parameters = fit.fit_results
    except:
        best_parameters = fit.MCMC_chains[np.argmax(fit.MCMC_chains[:, -1])]

    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(best_parameters)

    telescope_index = 0
    telescope_reference_name  =  fit.event.telescopes[telescope_index].name

    parameters_to_plot = copy.copy(best_parameters)

    # put all flux parameters to the telescope 0 scale

    for index, telescope in enumerate(fit.event.telescopes):

        telescope_name = telescope.name
        parameters_to_change = [(i, name) for i, name in enumerate(pyLIMA_parameters._fields) if telescope_name in name]

        for parameter in parameters_to_change:
            parameters_to_plot[parameter[0]] = getattr(pyLIMA_parameters, parameter[1].replace(telescope_name,
                                                                                               telescope_reference_name))
    for telescope_index,telescope in enumerate(fit.event.fake_telescopes):


        telescope_index_color = [i for i in range(len(fit.event.telescopes)) if
                       fit.event.telescopes[i].name == telescope.name][0]

        if telescope_index == 0:

            plot_model(figure_axes[0],fit, parameters =parameters_to_plot,  telescope_index = telescope_index,
               model_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index_color])

        else:
            plot_model(figure_axes[0], fit, parameters=parameters_to_plot, telescope_index=telescope_index,
                   model_color=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index_color], label=False)

    plot_align_data(figure_axes[0], fit, telescope_index=0, parameters = best_parameters)

    plot_residuals(figure_axes[1], fit, parameters=best_parameters)
    figure_axes[0].legend(numpoints=1, loc='best',
                      fancybox=True, framealpha=0.5)
    return figure



def initialize_plot_lightcurve(fit):
    """Initialize the lightcurve plot.

    :param object fit: a fit object. See the microlfits for more details.

    :return: a matplotlib figure  and the corresponding matplotlib axes
    :rtype: matplotlib_figure,matplotlib_axes

    """
    fig_size = [10, 10]
    figure, figure_axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                                       figsize=(fig_size[0], fig_size[1]), dpi=75)
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.99, wspace=0.2, hspace=0.1)
    figure_axes[0].grid()
    figure_axes[1].grid()
    # fig_size = plt.rcParams["figure.figsize"]
    figure.suptitle(fit.event.name, fontsize=30 * fig_size[0] / len(fit.event.name))

    figure_axes[0].set_ylabel('Mag', fontsize=5 * fig_size[1] * 3 / 4.0)
    figure_axes[0].yaxis.set_major_locator(MaxNLocator(4))
    figure_axes[0].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)

    figure_axes[0].text(0.01, 0.96, 'provided by pyLIMA', style='italic', fontsize=10,
                    transform=figure_axes[0].transAxes)

    figure_axes[1].set_xlabel('HJD', fontsize=5 * fig_size[0] * 3 / 4.0)
    figure_axes[1].xaxis.set_major_locator(MaxNLocator(3))
    figure_axes[1].yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))


    figure_axes[1].ticklabel_format(useOffset=False, style='plain')
    figure_axes[1].set_ylabel('Residuals', fontsize=5 * fig_size[1] * 2 / 4.0)
    figure_axes[1].tick_params(axis='x', labelsize=3.5 * fig_size[0] * 3 / 4.0)
    figure_axes[1].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)

    return figure, figure_axes


def initialize_plot_parameters(fit):
    """Initialize the parameters plot.

    :param object fit: a fit object. See the microlfits for more details.
    :return: a matplotlib figure  and the corresponding matplotlib axes.
    :rtype: matplotlib_figure,matplotlib_axes
    """

    fig_size = [10, 10]

    dimension_y = np.floor(len(fit.fits_result) / 3)
    dimension_x = len(fit.fits_result) - 3 * dimension_y

    figure, figure_axes = plt.subplots(dimension_x, dimension_y, figsize=(fig_size[0], fig_size[1]))
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.20, right=0.9, wspace=0.6, hspace=0.5)
    return figure, figure_axes


def model_magnitude(fit, parameters = None,  telescope_index = 0):
    """Compute the model in magnitude for arbitrary time stamps.

       :param object fit: a fit object. See the microlfits for more details.
       :param matplotlib_axes figure_axe: a matplotlib axes correpsonding to the figure.
       :param list parameters : a list of model parameters.
       :param np.array time : the time stamps for the model.
       :param int telescope_index : which telescope you want a model (depends on filter, location etc...)


    """


    reference_telescope = fit.event.fake_telescopes[telescope_index]



    return magnitude

def plot_model(figure_axe, fit, parameters = None, telescope_index = 0, model_color = 'b',model_alpha = 1.0, label=True):
    """Plot the microlensing model corresponding to parameters, time and with the same properties as  telescope,  the best fit and first telescope.

    :param object fit: a fit object. See the microlfits for more details.
    :param matplotlib_axes figure_axe: a matplotlib axes correpsonding to the figure.
    :param list parameters : a list of model parameters.
    :param np.array time : the time stamps for the model.
    :param int telescope_index : which telescope you want a model (depends on filter, location etc...)
    :param str model_color : the model color
    :param float model_alpha : the intensity of the model line

    """

    if parameters is not None:

        pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(parameters)

    else:

        pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(fit.fit_results)

    telescope = fit.event.fake_telescopes[telescope_index]

    time = telescope.lightcurve_flux[:,0]
    flux_model = fit.model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[0]
    magnitude = microltoolbox.flux_to_magnitude(flux_model)

    try:
        if label == True:

            figure_axe.plot(time, magnitude, c=model_color, label=fit.model.model_type, lw=3,alpha=model_alpha)


        else:

            figure_axe.plot(time, magnitude, c=model_color, lw=3,alpha=model_alpha)
    except:
        import pdb;
        pdb.set_trace()
    if telescope_index == 0:
        figure_axe.set_ylim(
            [min(magnitude) - plot_lightcurve_windows, max(magnitude) + plot_lightcurve_windows])
        figure_axe.set_xlim(
            [pyLIMA_parameters.to - 2 * np.abs(pyLIMA_parameters.tE),
             pyLIMA_parameters.to + 2 * np.abs(pyLIMA_parameters.tE) ])

        figure_axe.invert_yaxis()



def plot_residuals(figure_axe, fit, parameters =None ):
    """Plot the residuals from the fit.

    :param object fit: a fit object. See the microlfits for more details.
    :param matplotlib_axes figure_axe: a matplotlib axes correpsonding to the figure.
    """

    if parameters is not None:

        pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(parameters)

    else:
        pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(fit.fit_results)

    for index, telescope in enumerate(fit.event.telescopes):
        time = telescope.lightcurve_flux[:, 0]
        flux = telescope.lightcurve_flux[:, 1]
        err_mag = telescope.lightcurve_magnitude[:, 2]

        flux_model = fit.model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[0]

        residuals = 2.5 * np.log10(flux_model / flux)

        figure_axe.errorbar(time, residuals, yerr=err_mag, ls='None', markersize=7.5,
                            marker=str(MARKER_SYMBOLS[0][index]), capsize=0.0)
    figure_axe.set_ylim([-plot_residuals_windows, plot_residuals_windows])
    figure_axe.invert_yaxis()
    figure_axe.yaxis.get_major_ticks()[-1].draw = lambda *args: None





def plot_align_data(figure_axe, fit, telescope_index = 0, parameters = None):
    """Plot the aligned data.

    :param matplotlib_axes figure_axe: a matplotlib axes correpsonding to the figure.
    :param object fit: a fit object. See the microlfits for more details.
    :param int telescope_index : the telescope to align data to.

    """



    normalised_lightcurves = microltoolbox.align_the_data_to_the_reference_telescope(fit, telescope_index, parameters)

    for index, telescope in enumerate(fit.event.telescopes):
        lightcurve = normalised_lightcurves[index]

        figure_axe.errorbar(lightcurve[:, 0], lightcurve[:, 1], yerr=lightcurve[:, 2], ls='None',
                            marker=str(MARKER_SYMBOLS[0][index]), markersize=7.5, capsize=0.0,
                            label=telescope.name)




def align_telescope_lightcurve(lightcurve_telescope_flux, model_ghost, model_telescope):
    """Align data to the survey telescope (i.e telescope 0).

    :param array_like lightcurve_telescope_mag: the survey telescope in magnitude
    :param float fs_reference: thce survey telescope reference source flux (i.e the fitted value)
    :param float g_reference: the survey telescope reference blending parameter (i.e the fitted
    value)
    :param float fs_telescope: the telescope source flux (i.e the fitted value)
    :param float g_reference: the telescope blending parameter (i.e the fitted value)

    :return: the aligned to survey lightcurve in magnitude
    :rtype: array_like
    """
    time = lightcurve_telescope_flux[:, 0]
    flux = lightcurve_telescope_flux[:, 1]
    error_flux = lightcurve_telescope_flux[:, 2]
    err_mag = microltoolbox.error_flux_to_error_magnitude(error_flux, flux)

    residuals = 2.5 * np.log10(model_telescope / flux)

    magnitude_normalised = microltoolbox.flux_to_magnitude(model_ghost) + residuals

    lightcurve_normalised = [time, magnitude_normalised, err_mag]

    lightcurve_mag_normalised = np.array(lightcurve_normalised).T

    return lightcurve_mag_normalised


def parameters_table(fit):
    """Plot the fit parameters and errors.
        :param object fit: a fit object. See the microlfits for more details.
        :param list best_parameters: a list containing the model you want to plot the trajectory
    """

    column_labels = ['Parameters', 'Errors']

    try:
        #DE or LM
        table_val = [fit.fit_results, (fit.fit_covariance.diagonal() ** 0.5).tolist() + [0.0]]
        raw_labels = list(fit.model.model_dictionnary.keys()) + ['Chi^2']


    except:
        # MCMC
        best_chain = np.argmax(fit.MCMC_chains[:,-1])

        table_val = [(fit.MCMC_chains[best_chain,:-1]).tolist()+[fit.MCMC_chains[best_chain,-1]*-2],
                     ((np.percentile(fit.MCMC_chains[:,:-1],84,axis=0)- np.percentile(fit.MCMC_chains[:,:-1],16,axis=0))/2).tolist()
                     + [0.0]]

        raw_labels = list(fit.model.model_dictionnary.keys())[:len(table_val[0])-1] + ['Chi^2']



    table_val = np.round(table_val, 5).tolist()
    table_val = np.array(table_val).T.tolist()

    table_colors = []
    raw_colors = []
    last_color = 'dodgerblue'
    for i in range(len(table_val)):
        table_colors.append([last_color, last_color])
        raw_colors.append(last_color)

        if last_color == 'dodgerblue':

            last_color = 'white'

        else:

            last_color = 'dodgerblue'

    fig_size = [10, 7.5]
    figure_table = plt.figure(figsize=(fig_size[0], fig_size[1]), dpi=75)
    table_axes = figure_table.add_subplot(111, aspect=1)



    the_table = table_axes.table(cellText=table_val, cellColours=table_colors, rowColours=raw_colors,
                                 rowLabels=raw_labels,
                                 colLabels=column_labels, loc='center left',
                                 rowLoc='left', colLoc='center',
                                 bbox=[0.0, -0.0, 1.0, 1.0]
                                 )
    table_axes.get_yaxis().set_visible(False)
    table_axes.get_xaxis().set_visible(False)

    for (row, col), cell in the_table.get_celld().items():

        if (row == 0) or (col == -1):

            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    for cell in the_table._cells:
        the_table._cells[cell].set_alpha(.5)

    title = fit.model.event.name + ' : ' + fit.model.model_type
    figure_table.suptitle(title, fontsize=30 * fig_size[0] / len(title))
    return figure_table

#################################
#import pdb; pdb.set_trace()
################################
def plot_geometry(fit):
    """Plot the lensing geometry (i.e source trajectory) and the table of best parameters.
    :param object fit: a fit object. See the microlfits for more details.
    :param list best_parameters: a list containing the model you want to plot the trajectory
    """


    if len(fit.fit_results) != 0:
        best_parameters = fit.fit_results
    else:
        best_parameters = fit.MCMC_chains[np.argmax(fit.MCMC_chains[:,-1])]

    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(best_parameters)

    fig_size = [10, 10]
    figure_trajectory = plt.figure(figsize=(fig_size[0], fig_size[1]), dpi=75)

    figure_axes = figure_trajectory.add_subplot(111, aspect=1)
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)


    for telescope in fit.event.fake_telescopes:

        platform = 'Earth'
        if telescope.location == 'Space':

            platform = telescope.name

        reference_telescope = telescope

        telescope_index = [i for i in range(len(fit.event.telescopes)) if
                           fit.event.telescopes[i].name == telescope.name][0]

        fit.model.find_origin(pyLIMA_parameters)
        to, uo = fit.model.uo_to_from_uc_tc(pyLIMA_parameters)

        trajectory_x, trajectory_y, separation = fit.model.source_trajectory(reference_telescope,
                                                                             to, uo,
                                                                             pyLIMA_parameters.tE,
                                                                             pyLIMA_parameters)

        figure_axes.plot(trajectory_x, trajectory_y,
                         c=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
                         label=platform)


        for index in [-1,0,1]:

            try:
                index = np.argmin(np.abs(telescope.lightcurve_magnitude[:,0]-
                                         (pyLIMA_parameters.to+index*pyLIMA_parameters.tE)))
                derivative = (trajectory_y[index-1]-trajectory_y[index+1])/(trajectory_x[index-1]-trajectory_x[index+1])

                figure_axes.annotate('',xy=(trajectory_x[index], trajectory_y[index]),
                                             xytext=(trajectory_x[index]-0.001, trajectory_y[index]-0.001*derivative),
                                            arrowprops=dict(arrowstyle="->",mutation_scale =35,
                                                            color= plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index]))

            except:
                pass
        if fit.model.model_type == 'DSPL':
            trajectory_x, trajectory_y,separation = fit.model.source_trajectory(reference_telescope,
                                                             pyLIMA_parameters.to + pyLIMA_parameters.delta_to,
                                                             pyLIMA_parameters.uo + pyLIMA_parameters.delta_uo,
                                                             pyLIMA_parameters.tE,
                                                             pyLIMA_parameters)

            figure_axes.plot(trajectory_x, trajectory_y,
                             c=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],alpha=0.5)




    if 'BL' in fit.model.model_type:
        regime, caustics, cc = microlcaustics.find_2_lenses_caustics_and_critical_curves(10 ** pyLIMA_parameters.logs,
                                                                                         10 ** pyLIMA_parameters.logq,
                                                                                         resolution=5000)
        for count, caustic in enumerate(caustics):

            try:
                figure_axes.plot(caustic.real, caustic.imag, lw=3, c='r')
                figure_axes.plot(cc[count].real, cc[count].imag, '--k')

            except AttributeError:
                pass

    else:

        figure_axes.scatter(0, 0, s=10, c='r')
        einstein_ring = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--')
        figure_axes.add_artist(einstein_ring)


    for telescope_index,telescope in enumerate(fit.event.telescopes):

        fit.model.find_origin(pyLIMA_parameters)
        to, uo = fit.model.uo_to_from_uc_tc(pyLIMA_parameters)

        trajectory_x, trajectory_y, separation = fit.model.source_trajectory(telescope,
                                                                             to, uo,
                                                                             pyLIMA_parameters.tE,
                                                                             pyLIMA_parameters)

        if 'rho' in pyLIMA_parameters._fields:

            rho = pyLIMA_parameters.rho
        else:
            rho = 10**-3

        patches = [plt.Circle((x,y), rho, color = plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
                                                              fill=None) for x,y in zip(trajectory_x, trajectory_y)]
        coll = matplotlib.collections.PatchCollection(patches,match_original=True)

        figure_axes.scatter(trajectory_x, trajectory_y, edgecolors =  plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
                                                      facecolors= 'none',label=telescope.name,s=0.1)

        figure_axes.add_collection(coll)

    legend = figure_axes.legend(numpoints=1,  loc='best',fancybox=True,framealpha = 0.5)
    for handle in legend.legendHandles:
        try:
            handle.set_sizes([100])
        except:
            pass

    figure_axes.xaxis.set_major_locator(MaxNLocator(5))
    figure_axes.yaxis.set_major_locator(MaxNLocator(5))
    figure_axes.xaxis.get_major_ticks()[0].draw = lambda *args: None
    figure_axes.yaxis.get_major_ticks()[0].draw = lambda *args: None
    figure_axes.xaxis.get_major_ticks()[-1].draw = lambda *args: None
    figure_axes.yaxis.get_major_ticks()[-1].draw = lambda *args: None

    figure_axes.set_xlabel(r'$x(\theta_E)$', fontsize=25)
    figure_axes.set_ylabel(r'$y(\theta_E)$', fontsize=25)
    figure_axes.tick_params(axis='x', labelsize=15)
    figure_axes.tick_params(axis='y', labelsize=15)

    figure_axes.axis([-3,3,-3,3])
    title = fit.model.event.name + ' : ' + fit.model.model_type
    figure_trajectory.suptitle(title, fontsize=30 * fig_size[0] / len(title))
    return figure_trajectory

def plot_detection_zone_area(fit): 
    """Plot the lensing geometry (i.e source trajectory), image track and simulate planets.
    :param object fit: a fit object. See the microlfits for more details.
    :param list best_parameters: a list containing the model you want to plot the trajectory
    """


    if len(fit.fit_results) != 0:
        best_parameters = fit.fit_results
    else:
        best_parameters = fit.MCMC_chains[np.argmax(fit.MCMC_chains[:,-1])]

    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(best_parameters)

    fig_size = [10, 11]
    figure_trajectory = plt.figure(figsize=(fig_size[0], fig_size[1]), dpi=75)

    figure_axes = figure_trajectory.add_subplot(111, aspect=1)
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    
    #parameters
    
    your_event = fit.event
    te = pyLIMA_parameters.tE
    t0, u0 = fit.model.uo_to_from_uc_tc(pyLIMA_parameters)
    t0_err = np.sqrt(your_event.fits[-1].fit_covariance[0][0])
    u0_err = np.sqrt(your_event.fits[-1].fit_covariance[1][1])
    te_err = np.sqrt(your_event.fits[-1].fit_covariance[2][2])
    rs=1e-3
    
    x = []
    y = []
    chi1 = []
    data_time = []
    time = []
    flux = []
    fs = []
    fs_err = []
    fb_err = []
    fb = []
    ferr = []
    flux_err = []
    txplus = []
    txminus = []
    typlus = []
    tyminus = []
    q_sim = []
    d_sim = []
    d_planet = []
    q_planet = []
    qErr = []
    teErr = []
    x_chi2 = []
    
    for index,telescope in enumerate(your_event.telescopes):
        l = index+1
        data_time_tel = list(your_event.fits[-1].event.telescopes[l-1].lightcurve_magnitude[:,0])
        data_time.append(data_time_tel)
        time += data_time_tel
        flux_tel = list(fit.event.telescopes[l-1].lightcurve_flux[:,1])
        flux += flux_tel
        fs_tel =  fit.fit_results[2*l+1]
        fb_tel = fs_tel *  fit.fit_results[2*l+2]
        ferr_tel =  list(fit.event.telescopes[l-1].lightcurve_flux[:,2])
        fs.append(fs_tel)
        fb.append(fb_tel)
        ferr.append(ferr_tel)
        flux_err += ferr_tel
        fsErr = [(fit.fit_covariance.diagonal() ** 0.5).tolist()[-l]]
        fbErr = [(fit.fit_covariance.diagonal() ** 0.5).tolist()[-index]]
        fs_err += fsErr
        fb_err += fbErr
        txplus_tel = []
        txminus_tel = []
        typlus_tel = []
        tyminus_tel = []
        for t in data_time_tel:
            pspl,uplus,uminus,prox,proy,muplus,muminus,spplus,spminus = detection_zone_area.amp_and_imagesf(t,u0,te,t0)
            txplus_tel.append(prox*uplus)
            typlus_tel.append(proy*uplus)
            txminus_tel.append(prox*uminus)
            tyminus_tel.append(proy*uminus)
        txplus.append(txplus_tel)
        txminus.append(txminus_tel)
        typlus.append(typlus_tel)
        tyminus.append(tyminus_tel)
        figure_axes.scatter(txplus_tel,typlus_tel,color='navy', linewidths='0.0001', marker='.', alpha = 0.6, label = telescope.name)
        figure_axes.scatter(txminus_tel,tyminus_tel,color='navy', linewidths='0.0001', marker='.', alpha = 0.6)
   #simulate planets
    #number of simulations:
    sim = 1000
    #threshold for sum of chi2:
    thr = 100
    dsim = detection_zone_area.suzukian_s.rvs(size=sim)
    bar = pyprind.ProgBar((sim),monitor=True)
    for i in range(sim):
        #use simulate_simple to simulate xp and yp (the planet's coordinates)
        #q is simulated via the Suzukian Mass Ratio Function
        #d gets calculated using sphere point picking and Suzuki
        
        #xp,yp,q1,rs,a,b,dout,c,multiflag,bdflag,ffpflag = detection_zone_area.simulate_simple(te)
        q = np.random.choice([float(detection_zone_area.suzukian_q_right.rvs(size=1)), float(detection_zone_area.suzukian_q_left.rvs(size=1))], p = (detection_zone_area.right, 1-detection_zone_area.right))
        q_sim.append(q)
        
        d_s = dsim[i]
        d_sim.append(d_s)
        
        xp,yp = detection_zone_area.simulate_planet_pos(d_s)
        idxclose_all = []
        d = {}
        for index,telescope in enumerate(your_event.telescopes):
            #figure out the closest light curve data point for each planet and calculate delta chi squared for those ones
            #check up to 7 of the planet's nearest data points to be over the chi^2 threshold
            deltarplus=(np.array(txplus[index])-xp)**2+(np.array(typlus[index])-yp)**2
            deltarminus=(np.array(txminus[index])-xp)**2+(np.array(tyminus[index])-yp)**2
            idxplus=detection_zone_area.find_idx_nearest(deltarplus,0.0)
            idxminus=detection_zone_area.find_idx_nearest(deltarminus,0.0)
            if deltarplus[idxplus]<=deltarminus[idxminus]:
                idxclose=idxplus
            else:
                idxclose=idxminus
            idxclose_all.append(deltarplus[idxplus])
            idxclose_all.append(deltarminus[idxminus])
            d[str(deltarplus[idxplus])] = [idxclose, index]
            d[str(deltarminus[idxminus])] = [idxclose, index]
        idxclose1 = d[str(min(idxclose_all))][0]
        index_tel = d[str(min(idxclose_all))][1]
        t = data_time[index_tel][idxclose1]
 
        delta_chisqr1 = thr +1
        delta_chisqr2 = thr +1
        delta_chisqr3 = thr +1
        delta_chisqr1m = thr +1
        delta_chisqr2m = thr +1
        delta_chisqr3m = thr +1
        if idxclose1+1 < len(data_time[index_tel]):
            delta_chisqr1 = detection_zone_area.reldev(q,t,u0,te,t0,fs[index_tel],fb[index_tel],ferr[index_tel][idxclose1+1],rs,xp,yp,d_s)[0]
        if idxclose1+2 < len(data_time[index_tel]):
            delta_chisqr2 = detection_zone_area.reldev(q,t,u0,te,t0,fs[index_tel],fb[index_tel],ferr[index_tel][idxclose1+2],rs,xp,yp,d_s)[0]
        if idxclose1+3 < len(data_time[index_tel]):
            delta_chisqr3 = detection_zone_area.reldev(q,t,u0,te,t0,fs[index_tel],fb[index_tel],ferr[index_tel][idxclose1+3],rs,xp,yp,d_s)[0]
        if idxclose1-1 >= 0:
            delta_chisqr1m = detection_zone_area.reldev(q,t,u0,te,t0,fs[index_tel],fb[index_tel],ferr[index_tel][idxclose1-1],rs,xp,yp,d_s)[0]
        if idxclose1-2 >= 0:
            delta_chisqr2m = detection_zone_area.reldev(q,t,u0,te,t0,fs[index_tel],fb[index_tel],ferr[index_tel][idxclose1-2],rs,xp,yp,d_s)[0]
        if idxclose1-3 >= 0:
            delta_chisqr3m = detection_zone_area.reldev(q,t,u0,te,t0,fs[index_tel],fb[index_tel],ferr[index_tel][idxclose1-3],rs,xp,yp,d_s)[0]
        
        sorted_time = sorted(data_time)
        if idxclose1+1 >= len(data_time[index_tel]):
            if data_time[index_tel][0] != max(data_time)[0]:
                i = sorted_time.index(data_time[index_tel])+1 
                idx = data_time.index(sorted_time[i])
                delta_chisqr1 = detection_zone_area.reldev(q,t,u0,te,t0,fs[idx],fb[idx],ferr[idx][0],rs,xp,yp,d_s)[0]
        if idxclose1+2 >= len(data_time[index_tel]):
            if data_time[index_tel][0] != max(data_time)[0]:
                i = sorted_time.index(data_time[index_tel])+1 
                idx = data_time.index(sorted_time[i])
                try:
                    delta_chisqr2 = detection_zone_area.reldev(q,t,u0,te,t0,fs[idx],fb[idx],ferr[idx][1],rs,xp,yp,d_s)[0]
                except: 
                    pass
        if idxclose1+3 >= len(data_time[index_tel]):
            if data_time[index_tel][0] != max(data_time)[0]:
                i = sorted_time.index(data_time[index_tel])+1 
                idx = data_time.index(sorted_time[i])
                try:
                    delta_chisqr3 = detection_zone_area.reldev(q,t,u0,te,t0,fs[idx],fb[idx],ferr[idx][2],rs,xp,yp,d_s)[0]
                except: 
                    pass
        if idxclose1-1 < 0:
            if data_time[index_tel][0] != min(data_time)[0]:
                i = sorted_time.index(data_time[index_tel])-1 
                idx = data_time.index(sorted_time[i]) 
                delta_chisqr1m = detection_zone_area.reldev(q,t,u0,te,t0,fs[idx],fb[idx],ferr[idx][-1],rs,xp,yp,d_s)[0]

        if idxclose1-2 < 0:
            if data_time[index_tel][0] != min(data_time)[0]:
                i = sorted_time.index(data_time[index_tel])-1 
                idx = data_time.index(sorted_time[i]) 
                try:
                    delta_chisqr2m = detection_zone_area.reldev(q,t,u0,te,t0,fs[idx],fb[idx],ferr[idx][-2],rs,xp,yp,d_s)[0]
                except: 
                    pass
        if idxclose1-3 < 0:
            if data_time[index_tel][0] != min(data_time)[0]:
                i = sorted_time.index(data_time[index_tel])-1 
                idx = data_time.index(sorted_time[i]) 
                try:
                    delta_chisqr3m = detection_zone_area.reldev(q,t,u0,te,t0,fs[idx],fb[idx],ferr[idx][-3],rs,xp,yp,d_s)[0]
                except: 
                    pass
                            
        chi2 = detection_zone_area.reldev(q,t,u0,te,t0,fs[index_tel],fb[index_tel],ferr[index_tel][idxclose1],rs,xp,yp,d_s)[0]
        
        if chi2 > thr and delta_chisqr1 > thr and delta_chisqr2 > thr and delta_chisqr3 > thr and delta_chisqr1m > thr and delta_chisqr2m > thr and delta_chisqr3m > thr:
            #only accept planets beyond the set threshold
            x_chi2.append(xp)
            #Fisher Matrix
            Fisher_inv = detection_zone_area.fisher(time,q,u0,te,t0,xp,yp,fs[index_tel],fb[index_tel],flux,fs_err[index_tel],fb_err[index_tel],flux_err,d_s)
            q_err = np.sqrt(Fisher_inv[0][0])
            te_err = np.sqrt(Fisher_inv[1][1])
            qErr.append(q_err)
            teErr.append(te_err)
            if q_err < 0.2*q and te_err < 0.2*te:
                x.append(xp)
                y.append(yp)
                chi1.append(chi2)
                d_planet.append(d_s)
                q_planet.append(q)
        bar.update()

    t_arr = np.linspace(t0-4.*te,t0+4.*te,5000)
    images = detection_zone_area.generate_image_arrays(t_arr,u0,te,t0,u0_err,te_err,t0_err)
    figure_axes.plot(images[:,0],images[:,1], alpha = 0.5, color='navy', label = 'image track')
    figure_axes.plot(images[:,2],images[:,3], alpha = 0.5, color='navy')
    figure_axes.plot(images[:,4],images[:,5],color='cornflowerblue', alpha = 0.5)
    figure_axes.plot(images[:,6],images[:,7],color='cornflowerblue', alpha = 0.5)
    figure_axes.plot(images[:,8],images[:,9],color='cornflowerblue', alpha = 0.5)
    figure_axes.plot(images[:,10],images[:,11],color='cornflowerblue', alpha = 0.5)
    figure_axes.plot(images[:,12],images[:,13], alpha = 0.5, color='navy')
    figure_axes.plot(images[:,14],images[:,15],color='cornflowerblue', alpha = 0.5)
    figure_axes.plot(images[:,16],images[:,17],color='cornflowerblue', alpha = 0.5)
    figure_axes.scatter(x, y, c = chi1, s = 8, alpha = 1, cmap='autumn')
    try:
        plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min(chi1),vmax=max(chi1)), cmap='autumn'), fraction = 0.046)
    except: 
        pass
    figure_axes.text(3.32,-3.36, r'$\Delta$' + ' ' + r'$\chi^2$')
    figure_axes.text(-3,-4.38, 'Number of simulations: ' + str(sim) + '\nNumber of planets beyond the threshold of ' + r'$\chi^2=$' + str(thr) + ': ' + str(len(x_chi2)) + '\nNumber of planets that passed the ' + r'$\chi^2$ ' + 'threshold\nand have te and q uncertainties below 20%' + ': '+ str(len(x)))
    
    #save data for the accepted planets in a txt-file 
    data = np.column_stack((x,y,q_planet,d_planet,chi1))
    header = 'x-coordinate y-coordinate mass_ratio separation chi_squared'
#    np.savetxt('simulated_planets.txt', data, header = header)
    
    for telescope in fit.event.fake_telescopes:

        platform = 'Earth'
        if telescope.location == 'Space':

            platform = telescope.name

        reference_telescope = telescope

        telescope_index = [i for i in range(len(fit.event.telescopes)) if
                           fit.event.telescopes[i].name == telescope.name][0]

        fit.model.find_origin(pyLIMA_parameters)
        to, uo = fit.model.uo_to_from_uc_tc(pyLIMA_parameters)

        trajectory_x, trajectory_y, separation = fit.model.source_trajectory(reference_telescope,
                                                                             to, uo,
                                                                             pyLIMA_parameters.tE,
                                                                             pyLIMA_parameters)
        
        figure_axes.plot(trajectory_x, trajectory_y, color = 'navy', label = platform)
#                         c=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
#                         label=platform)


        for index in [-1,0,1]:

            try:
                index = np.argmin(np.abs(telescope.lightcurve_magnitude[:,0]-
                                         (pyLIMA_parameters.to+index*pyLIMA_parameters.tE)))
                derivative = (trajectory_y[index-1]-trajectory_y[index+1])/(trajectory_x[index-1]-trajectory_x[index+1])

                figure_axes.annotate('',xy=(trajectory_x[index], trajectory_y[index]),
                                             xytext=(trajectory_x[index]-0.001, trajectory_y[index]-0.001*derivative),
                                            arrowprops=dict(arrowstyle="->",mutation_scale =35, color = 'navy'))
#                                                            color= plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index]))

            except:
                pass
        if fit.model.model_type == 'DSPL':
            trajectory_x, trajectory_y,separation = fit.model.source_trajectory(reference_telescope,
                                                             pyLIMA_parameters.to + pyLIMA_parameters.delta_to,
                                                             pyLIMA_parameters.uo + pyLIMA_parameters.delta_uo,
                                                             pyLIMA_parameters.tE,
                                                             pyLIMA_parameters)

            figure_axes.plot(trajectory_x, trajectory_y,
                             c=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],alpha=0.5)




#    if 'BL' in fit.model.model_type:
#        regime, caustics, cc = microlcaustics.find_2_lenses_caustics_and_critical_curves(10 ** pyLIMA_parameters.logs,
#                                                                                         10 ** pyLIMA_parameters.logq,
#                                                                                         resolution=5000)
#        for count, caustic in enumerate(caustics):

#            try:
#                figure_axes.plot(caustic.real, caustic.imag, lw=3, c='r')
#                figure_axes.plot(cc[count].real, cc[count].imag, '--k')

#            except AttributeError:
#                pass

#    else:

        figure_axes.scatter(0, 0, s=10, c='k')
        einstein_ring = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--')
        figure_axes.add_artist(einstein_ring)


    for telescope_index,telescope in enumerate(fit.event.telescopes):

        fit.model.find_origin(pyLIMA_parameters)
        to, uo = fit.model.uo_to_from_uc_tc(pyLIMA_parameters)

        trajectory_x, trajectory_y, separation = fit.model.source_trajectory(telescope,
                                                                             to, uo,
                                                                             pyLIMA_parameters.tE,
                                                                             pyLIMA_parameters)

        if 'rho' in pyLIMA_parameters._fields:

            rho = pyLIMA_parameters.rho
        else:
            rho = 10**-3

        patches = [plt.Circle((x,y), rho, color = plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
                                                              fill=None) for x,y in zip(trajectory_x, trajectory_y)]
        coll = matplotlib.collections.PatchCollection(patches,match_original=True)

        figure_axes.scatter(trajectory_x, trajectory_y, edgecolors =  plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
                                                      facecolors= 'none',s=0.1)

        figure_axes.add_collection(coll)

    legend = figure_axes.legend(numpoints=1,  loc='best',fancybox=True,framealpha = 0.5)
    for handle in legend.legendHandles:
        try:
            handle.set_sizes([100])
        except:
            pass

    figure_axes.xaxis.set_major_locator(MaxNLocator(5))
    figure_axes.yaxis.set_major_locator(MaxNLocator(5))
    figure_axes.xaxis.get_major_ticks()[0].draw = lambda *args: None
    figure_axes.yaxis.get_major_ticks()[0].draw = lambda *args: None
    figure_axes.xaxis.get_major_ticks()[-1].draw = lambda *args: None
    figure_axes.yaxis.get_major_ticks()[-1].draw = lambda *args: None

    figure_axes.set_xlabel(r'$x(\theta_E)$', fontsize=25)
    figure_axes.set_ylabel(r'$y(\theta_E)$', fontsize=25)
    figure_axes.tick_params(axis='x', labelsize=15)
    figure_axes.tick_params(axis='y', labelsize=15)

    figure_axes.axis([-3,3,-3,3])
    title = 'Planet Simulation'
    figure_trajectory.suptitle(title, fontsize=30 * fig_size[0] / len(title))
    
    #plot histogramms for q and d
    fig_size = [10, 10]
    figure_distributions, axes = plt.subplots(2, 2, figsize=(fig_size[0], fig_size[1]))
    axes[0][0].hist(q_sim, bins=100,color = 'cornflowerblue', histtype = 'bar', label = 'simulated mass ratio')
    axes[0][0].hist(q_planet, bins=100, color = 'navy', histtype = 'bar', label = 'mass ratio of accepted planets')
    axes[0][0].set_xlabel('q', fontsize=15)
    #axes[0][0].set_yscale('log')
    #axes[0][0].set_xscale('log')
    
    axes[1][0].hist(d_sim, bins=100, color = 'cornflowerblue', histtype = 'bar', label = 'simulated separation')
    axes[1][0].hist(d_planet, bins=100, color = 'navy', histtype = 'bar', label = 'separation of accepted planets')
    axes[1][0].set_xlabel('d', fontsize=15)
    
    axes[0][1].hist(qErr, bins=100, color = 'cornflowerblue', histtype = 'bar')
    axes[0][1].set_xlabel('q error', fontsize=15)
    axes[1][1].hist(teErr, bins=100, color = 'cornflowerblue', histtype = 'bar')
    axes[1][1].set_xlabel('te error', fontsize=15)
    
    figure_distributions.subplots_adjust(top=0.8, bottom=0.2, wspace=0.2, hspace=0.3)
    figure_distributions.suptitle('Distributions', fontsize=30 * fig_size[0] / len(title))
    axes[0][0].legend(numpoints=1,  loc='best',fancybox=True,framealpha = 0.5, fontsize = 'x-small',markerscale = '0.5')
    axes[1][0].legend(numpoints=1,  loc='best',fancybox=True,framealpha = 0.5, fontsize = 'x-small',markerscale = '0.5')

    return figure_trajectory, figure_distributions