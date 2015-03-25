
##Copyright (c) 2015 duncan g. smith
##
##Permission is hereby granted, free of charge, to any person obtaining a
##copy of this software and associated documentation files (the "Software"),
##to deal in the Software without restriction, including without limitation
##the rights to use, copy, modify, merge, publish, distribute, sublicense,
##and/or sell copies of the Software, and to permit persons to whom the
##Software is furnished to do so, subject to the following conditions:
##
##The above copyright notice and this permission notice shall be included
##in all copies or substantial portions of the Software.
##
##THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
##OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
##THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
##OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
##ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
##OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division

from collections import defaultdict
import csv
import random
import operator
import itertools

import numpy


############## utility functions ##############


def parse_file(filename, **kwargs):
    """
    A utility function that uses the I{csv} module to parse
    a file containing variable names and records. Returns the
    variable names and the records.

    @type  filename: C{str}
    @param filename: path to file to be parsed
    @rtype:          C{tuple}
    @return:         a list of headers and a list of records
    """
    with open(filename, 'rb') as f:
        reader = csv.reader(f, **kwargs)
        return next(reader), list(reader)

def write_file(filename, names, records, **kwargs):
    """
    A utility function that uses the I{csv} module to write
    a file containing variable names and records.

    @type  filename: C{str}
    @param filename: file path
    @type     names: C{list}
    @param    names: list of variable names
    @type   records: C{list}
    @param  records: list of records
    """
    with open(filename, 'wb') as f:
        writer = csv.writer(f, **kwargs)
        writer.writerow(names)
        for rec in records:
            writer.writerow(rec)

def aligned(records, indices):
    """
    Takes I{N} records and a list of indices. Returns a transposed
    (and possibly marginalised) list of records.

    @type  records: a sequence
    @param records: I{N} records (fixed length lists of values)
    @type  indices: C{list}
    @param indices: a list of indices that specify a transposition. e.g. The
                    value at index C{indices[0]} will appear at index 0 in
                    the transposed record
    @rtype:         C{list}
    @return:        I{N} match statuses (1 for a match, 0 otherwise)
    """
    new_recs = []
    for rec in records:
        new_recs.append([rec[i] for i in indices])
    return new_recs

def blocked(records, indices):
    """
    Takes records and indices for blocking. Returns
    a mapping of blocking values to a corresponding list of records.

    @type  records: a sequence
    @param records: records as tuples of values
    @type  indices: C{list}
    @param indices: indices of fields which are equal if a pair of records are a match
    @rtype:         C{defaultdict}
    @return:        mapping of tuples of values on blocked fields to corresponding records
    """
    res = defaultdict(list)
    for rec in records:
        key = tuple(rec[index] for index in indices)
        res[key].append(rec)
    return res

def simscores(pairs, keys, sim_funcs=None, missing=''):
    """
    Takes I{N} record pairs, I{n} indices for the key variables
    and I{n} similarity score functions; and returns an array of
    similarity scores. If I{sim_funcs} is C{None}, then comparison
    defaults to strict equality.

    Record pairs must be aligned on key variables.

    @type      pairs: a sequence
    @param     pairs: I{N} aligned record pairs
    @type       keys: a sequence
    @param      keys: I{n} indices of the key variables
    @type  sim_funcs: C{None} or C{list}
    @param sim_funcs: similarity functions corresponding to key variables
    @type    missing: C{object}
    @param   missing: object denoting a missing record value
    @rtype:           numpy array
    @return:          (I{N},I{n}) shaped array containing similarity scores
                      or -99999999 for missing values
    """
    scores = numpy.zeros((len(pairs), len(keys)), dtype=numpy.float64)
    for i, (a,b) in enumerate(pairs):
        for j, key in enumerate(keys):
            if (a[key] == missing) or (b[key] == missing):
                scores[i,j] = -99999999
            elif not sim_funcs:
                if a[key] == b[key]:
                    scores[i,j] = 1
            else:
                scores[i,j] = sim_funcs[j](a[key], b[key])
    return scores

def categorized(simscores, limits):
    """
    Takes an (N,n) shaped array containing similarity scores for I{N} record
    pairs and I{n} variables. Returns a (N,c,n) shaped array of 1s and 0s such that
    a 1 denotes the relevant similarity score interval. The intervals are specified
    in I{limits}. Care should be taken to ensure that a missing value (perhaps specified
    by a large negative integer) generates I{c} 0s. Intervals include their
    upper limits, but not their lower limits.

    @type  simscores: numpy array
    @param simscores: (N,n) shaped array of similarity scores
    @type     limits: C{list}
    @param    limits: a list containing a lower limit followed by I{c} upper limits
    @rtype:           numpy array
    @return:          (N,c,n) shaped data array
    """
    arrs = []
    for i, low in enumerate(limits[:-1]):
        high = limits[i+1]
        arr = numpy.where((low < simscores) & (simscores <= high), 1, 0)
        arrs.append(arr)
    shape = list(simscores.shape)
    shape.insert(1,1)
    return numpy.concatenate([arr.reshape(shape) for arr in arrs], axis=1)

def u_probs(data):
    """
    Takes a data array of shape (N,c,n) - I{N} record pairs, I{c} similarity
    score intervals, and I{n} variables. Returns an estimate for the
    I{u}-probabilities based on the assumption that all record pairs
    are non matches.

    @type    data: numpy array
    @param   data: (N,c,n) shaped data array
    @rtype:        numpy array
    @return:       (c,n) shaped array of estimated I{u}-probabilities
    """
    N, c, n = data.shape
    margin = data.sum(axis=0)
    return margin / margin.sum(axis=0)


############## record linkage functions ##############


def EM(data, u=None, p=None, u_start=None, m_start=None, p_start=None, u_prior=1, m_prior=1, p_prior=1, max_its=10000, tol=1e-12):
    """
    Takes a data array and optional arguments and uses Expectation
    Maximization to generate either maximum likelihood or maximum a posteriori
    parameter estimates. The data array has shape (N,c,n) - I{N} record pairs, I{c} similarity
    score categories, and I{n} variables.

    If C{u} or C{p} are provided, then these remain fixed.

    C{u_start}, C{m_start} and C{p_start} are starting values. Sensible
    defaults are used for any of these that are not provided.

    C{u_prior}, C{m_prior} and C{p_prior} are Dirichlet priors for maximum
    a posteriori estimation. The default values provide maximum likelihood
    estimation. If C{int}s or C{float}s are provided, then all Dirichlet parameters
    take the supplied value.

    A C{ValueError} is raised if the algorithm does not converge within C{max_its} iterations.

    @type         data: numpy array
    @param        data: (N,c,n) shaped array containing similarity scores
    @type            u: C{None} or numpy array
    @param           u: fixed (c,n) shaped array of I{u}-probabilities
    @type            p: C{None} or C{float}
    @param           p: fixed marginal probability of a correct match
    @type      u_start: C{None} or numpy array
    @param     u_start: (c,n) shaped array of I{u}-probabilities
    @type      m_start: C{None} or numpy array
    @param     m_start: (c,n) shaped array of I{m}-probabilities
    @type      p_start: C{None} or C{float}
    @param     p_start: marginal probability of a correct match
    @type      u_prior: C{int}, C{float} or numpy array
    @param     u_prior: Dirichlet prior for I{u}-probabilities
    @type      m_prior: C{int}, C{float} or numpy array
    @param     m_prior: Dirichlet prior for I{m}-probabilities
    @type      p_prior: C{int}, C{float} or numpy array
    @param     p_prior: Dirichlet prior for [1-p,p]
    @type      max_its: C{int}
    @param     max_its: maximum number of iterations
    @type          tol: C{float}
    @param         tol: results are returned when consecutive log likelihoods
                        differ by less than this value
    @rtype:             tuple
    @return:            estimates for I{u}, I{m} and I{p}; posterior probabilities of a correct match
                        and the log likelihoods
    @raises ValueError: if C{max_its} iterations are reached before convergence
    """
    N, c, n = data.shape
    fit_u = fit_p = False
    if u is None:
        fit_u = True
        # produce starting values
        if u_start is None:
            u = u_probs(data)
        else:
            # ensure normalisation
            u = u_start / u_start.sum(axis=0)
    # produce starting values
    if m_start is None:
        m = numpy.zeros((c,n))
        m[:] = 0.05/(c-1)
        m[-1,:] = 0.95
    else:
        # ensure normalisation
        m = m_start / m_start.sum(axis=0)
    if p is None:
        fit_p = True
        p = p_start or 1/N**0.5
        # avoid degenerate case of p == 1
        if p == 1:
            p = 0.99
    last_u = u
    last_m = m
    log_likelihoods = []
    try:
        p_u, p_m = p_prior
    except TypeError: # not an array
        p_u = p_m = p_prior
    # EM
    for i in range(max_its):
        # expectation
        _u = numpy.prod(numpy.prod(u**data, 1), 1) * (1-p)
        _m = numpy.prod(numpy.prod(m**data, 1), 1) * p
        div = _u + _m
        g_u = _u / div
        g_m = _m / div
        #
        log_likelihoods.append(numpy.log(div).sum())
        #
        if fit_u:
            u = numpy.sum(g_u.reshape((N,1,1))*data , 0) + (u_prior-1)
            u = u / u.sum(axis=0)
        m = numpy.sum(g_m.reshape((N,1,1))*data, 0) + (m_prior-1)
        # maximization
        m = m / m.sum(axis=0)
        if fit_p:
            p = (numpy.sum(g_m)+(p_m-1)) / (N+p_u+p_m-2)
        # check for convergence
        if i and log_likelihoods[-1] - log_likelihoods[-2] <= tol:
            return u, m, p, g_m, log_likelihoods
    raise ValueError('max_its iterations reached without convergence')

def average_frequencies(records, indices, missing=''):
    """
    Takes records and indices of the key variables and returns
    the reciprocals of the numbers of distinct variable values.
    Used in EpiLink record linkage.

    @type  records: a sequence
    @param records: records as tuples of values
    @type  indices: C{list}
    @param indices: indices of key variables
    @type  missing: C{object}
    @param missing: object denoting a missing record value
    @rtype:         numpy array
    @return:        reciprocals of numbers of distinct variable values
    """
    freqs = numpy.zeros((len(indices),))
    for i in range(len(indices)):
        index = indices[i]
        vals = set(rec[i] for rec in records)
        vals.discard(missing)
        freqs[i] = len(vals)
    return 1/freqs

def EpiLink(simscores, w=None, error_rates=None, av_freqs=None):
    """
    Takes an (N,n) shaped array containing similarity scores for I{N} record
    pairs and I{n} variables, plus optional arguments. Returns a (N,) shaped array of
    EpiLink scores.

    If I{w} contains weights, then the weights are used to calculate EpiLink scores.

    If I{w} is C{None}, then C{error_rates} and C{freqs} must be supplied so that
    weights can be calculated.

    Missing values are handled by using the mean weight for the relevant field.

    @type    simscores: numpy array
    @param   simscores: (N,n) shaped array of similarity scores
    @type            w: C{None} or numpy array
    @param           w: (n,) shaped array of weights
    @type  error_rates: C{None} or numpy array
    @param error_rates: (N,n) or (n,) shaped array of error rates
    @type     av_freqs: C{None} or numpy array
    @param    av_freqs: (n,) shaped array of variable sizes (numbers of distinct categories)
    @rtype:             numpy array
    @return:            (N,) shaped array of EpiLink scores
    """
    masked = numpy.ma.masked_array(simscores, mask=numpy.where(simscores < 0, 1, 0))
    means = numpy.mean(masked, axis=0)
    simscores = numpy.where(simscores < 0, means, simscores)
    if w is None:
        w = numpy.log2((1-error_rates)/av_freqs)
    return (w*simscores).sum(1)/w.sum()

def likelihood_weighted_post(simscores, u, m, p):
    """
    Takes the similarity scores for I{N} record pairs over I{n} variables
    and combines them with Fellegi-Sunter I{u} and I{m}-probabilities
    and the marginal probability of a correct match to generate
    pseudo-posterior probabilities of a correct match. Essentially, it
    performs linear interpolation on the likelihoods.

    C{u[1]} and C{m[1]} are the probabilities of equal values given
    non-matching and matching records respectively.

    @type  simscores: numpy array
    @param simscores: (N,n) shaped array containing similarity scores
    @type          u: numpy array
    @param         u: (2,n) shaped array of I{u}-probabilities
    @type          m: numpy array
    @param         m: (2,n) shaped array of I{m}-probabilities
    @type          p: C{float}
    @param         p: marginal probability of a correct match
    @rtype:           numpy array
    @return:          I{N} pseudo-posterior probabilities of a correct match 
    """
    N,n = simscores.shape
    arr = simscores*(m[1]/u[1]) + (1-simscores)*(m[0]/u[0])
    arr = numpy.where(simscores < 0, 1, arr)
    arr = numpy.prod(arr, axis=1)
    post = arr * (p/(1-p))
    return post/(1+post)
    
def log_likelihood_weighted_post(simscores, u, m, p):
    """
    Takes the similarity scores for I{N} record pairs over I{n} variables
    and combines them with Fellegi-Sunter I{u} and I{m}-probabilities
    and the marginal probability of a correct match to generate
    pseudo-posterior probabilities of a correct match. Essentially, it
    performs linear interpolation on the log likelihoods / match weights.

    C{u[1]} and C{m[1]} are the probabilities of equal values given
    non-matching and matching records respectively.

    @type  simscores: numpy array
    @param simscores: (N,n) shaped array containing similarity scores
    @type          u: numpy array
    @param         u: (2,n) shaped array of I{u}-probabilities
    @type          m: numpy array
    @param         m: (2,n) shaped array of I{m}-probabilities
    @type          p: C{float}
    @param         p: marginal probability of a correct match
    @rtype:           numpy array
    @return:          I{N} pseudo-posterior probabilities of a correct match 
    """
    N,n = simscores.shape
    arr = (m[1]/u[1])**simscores * (m[0]/u[0])**(1-simscores)
    arr = numpy.where(simscores < 0, 1, arr)
    arr = numpy.prod(arr, axis=1)
    post = arr * (p/(1-p))
    return post/(1+post)

def Winkler_weighted_post(simscores, u, m, p, c=9/2):
    """
    Takes the similarity scores for I{N} record pairs over I{n} variables
    and combines them with Fellegi-Sunter I{u} and I{m}-probabilities
    and the marginal probability of a correct match to generate
    pseudo-posterior probabilities of a correct match. The approach used
    is that described in:

    Winkler, W. E. (1990). String Comparator Metrics and Enhanced
    Decision Rules in the Fellegi-Sunter Model of Record Linkage.
    Proceedings of the Section on Survey Research Methods (American
    Statistical Association) pp.354-359

    C{u[1]} and C{m[1]} are the probabilities of equal values given
    non-matching and matching records respectively.

    @type  simscores: numpy array
    @param simscores: (N,n) shaped array containing similarity scores
    @type          u: numpy array
    @param         u: (2,n) shaped array of I{u}-probabilities
    @type          m: numpy array
    @param         m: (2,n) shaped array of I{m}-probabilities
    @type          p: C{float}
    @param         p: marginal probability of a correct match
    @type          c: C{float}
    @param         c: additional interpolation parameter
    @rtype:           numpy array
    @return:          I{N} pseudo-posterior probabilities of a correct match 
    """
    N,n = simscores.shape
    arr = (m[1]/u[1])**(1-(1-simscores)*c) * (m[0]/u[0])**((1-simscores)*c)
    arr = numpy.where(simscores < 0, 1, arr)
    arr = numpy.prod(arr, axis=1)
    post = arr * (p/(1-p))
    return post/(1+post)


############## outputs ##############


def matches(pairs, ID_index):
    """
    Takes I{N} record pairs and an index for a unique ID field. Returns
    an array of match statuses.

    @type     pairs: a sequence
    @param    pairs: I{N} record pairs aligned on C{ID_index}
    @type  ID_index: C{int}
    @param ID_index: index of ID field (which is equal iff records are a match)
    @rtype:          numpy array
    @return:         I{N} match statuses (1 for a match, 0 otherwise)
    """
    arr = numpy.zeros((len(pairs),))
    for i, (rec1, rec2) in enumerate(pairs):
        if rec1[ID_index] == rec2[ID_index]:
            arr[i] = 1
    return arr

def predictions(scores, cutoff, op=operator.gt):
    """
    Takes I{N} scores (e.g. posterior probabilities or match weights),
    cutoff and an operator. Returns an array of predictions.

    @type  scores: numpy array
    @param scores: I{N} similarity scores
    @type  cutoff: C{int} or C{float}
    @param cutoff: cutoff for classification
    @type      op: C{function}
    @param     op: function returning a C{bool}
    @rtype:        numpy array
    @return:       predicted values (1 for a match, 0 otherwise
                   if using the default function)
    """
    return numpy.where(op(scores, cutoff), 1, 0)

def classification_table(matches, predictions):
    """
    Takes I{N} match statuses and I{N} predictions. Returns
    a classification table.

    @type      matches: numpy array
    @param     matches: I{N} match statuses (1 for a match, 0 otherwise)
    @type  predictions: numpy array
    @param predictions: I{N} predictions (1 for a match, 0 otherwise)
    @rtype:             numpy array
    @return:            (2,2) shaped array containing true and false positives
                        in the first row, and false and true negatives in the second row
    """
    # [[TP, FP],
    #  [FN, TN]]
    if not matches.shape == predictions.shape:
        raise ValueError('matches and predictions must have equal lengths')
    tbl = numpy.zeros((2,2))
    tbl[0,0] = (matches * predictions).sum()
    tbl[0,1] = predictions.sum() - tbl[0,0]
    tbl[1,0] = matches.sum() - tbl[0,0]
    tbl[1,1] = len(matches) - tbl.sum()
    return tbl


############## plots ##############


def log_likelihood_plot(filename, log_likelihoods, axis=None, format='jpeg', dpi=300, **kwargs):
    """
    Creates and saves a plot of log likelihoods.

    Any keyword arguments in **kwargs are passed to
    the Matplotlib 'plot' function.

    @type         filename: C{str}
    @param        filename: location of saved file
    @type  log_likelihoods: sequence
    @param log_likelihoods: log likelihoods
    @type             axis: C{None} or C{list}
    @param            axis: Matplotlib axis
    """
    import pylab
    pylab.plot(range(len(log_likelihoods)), log_likelihoods, **kwargs)
    if axis:
        pylab.axis(axis)
    pylab.xlabel(r"$Iterations$")
    lab = pylab.ylabel(r"$\ln(f(x\vert\Phi))$")
    pylab.savefig(filename, format=format, dpi=dpi)
    pylab.clf()

def series(pairs, ID_index, post):
    """
    Takes pairs of records and corresponding posterior probabilities and
    returns a series suitable for plotting ROC and precision recall
    curves.

    Record pairs must be aligned on a unique ID field so that match statuses
    can be determined.

    @type     pairs: a sequence
    @param    pairs: aligned record pairs
    @type  ID_index: C{int}
    @param ID_index: index of ID field (which is equal iff records are a match)
    @type      post: a sequence
    @param     post: posterior match probabilities for I{pairs}
    @rtype:          C{list}
    @return:         (posterior probability, match status) pairs sorted from
                     largest to smallest posterior probability
    """
    tups = [(post[i], pair[0][ID_index] == pair[1][ID_index]) for i, pair in enumerate(pairs)]
    random.shuffle(tups)
    tups.sort(reverse=True)
    return [tup[1] for tup in tups]

def ROC_curve(filename, series, legends, loc='lower right',
              colours=None, axis=None, format='jpeg', dpi=300, **kwargs):
    """
    Creates and saves an ROC curve.

    Any keyword arguments in **kwargs are passed to
    the Matplotlib 'plot' function.

    @type  filename: C{str}
    @param filename: location of saved file
    @type    series: C{list}
    @param   series: (posterior probability, match status) pairs sorted from
                     largest to smallest posterior probability
    @type   legends: C{list}
    @param  legends: names for series
    @type       loc: C{str}
    @param      loc: Matplotlib location for legends
    @type   colours: C{None} or C{list}
    @param  colours: Matplotlib colours
    @type      axis: C{None} or C{list}
    @param     axis: Matplotlib axis
    """
    if not len(series) == len(legends):
        raise ValueError('Number of series does not match number of legends')
    import pylab
    pylab.hold(True)
    if axis:
        pylab.axis(axis)
    pylab.gca().set_color_cycle(colours)
    for values in series:
        x_ser = [0]
        y_ser = [0]
        positives = values.count(True)
        negatives = len(values) - positives
        tp = fp = 0
        for val in values:
            tp += val
            fp += 1-val
            x_ser.append(fp/negatives)
            y_ser.append(tp/positives)
        pylab.plot(x_ser, y_ser, **kwargs)
    pylab.plot([-0.02,1], [-0.02,1], '%s--' % 'k')
    pylab.xlabel('False positive rate')
    lab = pylab.ylabel('True positive rate')
    pylab.legend(legends, loc=loc)
    pylab.savefig(filename, format=format, dpi=dpi)
    pylab.clf()

def precision_recall_curve(filename, series, legends, loc='lower left',
                           colours=None, axis=None, format='jpeg', dpi=300, **kwargs):
    """
    Creates and saves a precision recall curve.

    Any keyword arguments in **kwargs are passed to
    the Matplotlib 'plot' function.

    @type  filename: C{str}
    @param filename: location of saved file
    @type    series: C{list}
    @param   series: (posterior probability, match status) pairs sorted from
                     largest to smallest posterior probability
    @type   legends: C{list}
    @param  legends: names for series
    @type       loc: C{str}
    @param      loc: Matplotlib location for legends
    @type   colours: C{None} or C{list}
    @param  colours: Matplotlib colours
    @type      axis: C{None} or C{list}
    @param     axis: Matplotlib axis
    """
    if not len(series) == len(legends):
        raise ValueError('Number of series does not match number of legends')
    import pylab
    pylab.hold(True)
    if axis:
        pylab.axis(axis)
    pylab.gca().set_color_cycle(colours)
    for values in series:
        x_ser = [0]
        y_ser = [1]
        positives = values.count(True)
        negatives = len(values) - positives
        tp = fp = 0
        for val in values:
            tp += val
            fp += 1-val
            x_ser.append(tp/positives)
            y_ser.append(tp/(tp+fp))
        pylab.plot(x_ser, y_ser, **kwargs)
    pylab.xlabel('Recall')
    lab = pylab.ylabel('Precision')
    pylab.legend(legends, loc=loc)
    pylab.savefig(filename, format=format, dpi=dpi)
    pylab.clf()

