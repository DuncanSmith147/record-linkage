
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

from heapq import heappush, heappop

import numpy

"""
Decision rules for Fellegi-Sunter record linkage:

Fellegi, I.P. and Sunter A.B. (1969) A theory for record
linkage, JASA Vol. 64, No. 238, pp.1183-1210

In the following docstrings A1, A2 and A3 are the sets of correct matches,
uncertain matches and incorrect matches respectively.

We have two (conditional) error rates:

mu = the probability of allocation to A1 given an incorrect match
lambda = the probability of allocation to A3 given a correct match
"""

def error_rates_from_thresholds(u, m, p, probs, b=None):
    """
    Takes (c,n) shaped arrays I{u} and I{m}, the marginal
    probability of a correct match I{p}, and a list of thresholds
    on the posterior probability of a correct match. Returns
    the error rates corresponding to the thresholds.

    If supplied, C{b} is a mapping of variable indices to indices
    of levels to be used for blocking (generally the highest
    similarity score level).

    C{probs} must be sorted.

    @type      u: numpy array
    @param     u: (c,n) shaped array of I{u}-probabilities
    @type      m: numpy array
    @param     m: (c,n) shaped array of I{m}-probabilities
    @type      p: C{float}
    @param     p: marginal probability of a correct match
    @type  probs: C{list}
    @param probs: list of thresholds on the posterior
                  probability of a correct match
    @type      b: C{None} or C{dict}
    @param     b: mapping of variable indices to blocking levels
    @rtype:       C{list}
    @return:      tuples containing (mu, lambda) for each threshold
    @raises:      ValueError if C{probs} is not sorted
    """
    # does not deal with Pu or Pm (floating point issues
    # would make it unreliable)
    if not sorted(probs) == probs:
        raise ValueError('probs must be sorted')
    res = []
    lists, m_sum = _get_lists(u, m, b)
    mus = []
    lambdas = []
    likelihoods = [prob*(1-p)/(1-prob)/p for prob in probs]
    u_sum = 0
    gen = _gen_sorted_data(lists, desc=True)
    ratio = float('inf')
    for likelihood in reversed(likelihoods):
        while likelihood < ratio:
            ratio, arr = next(gen)
            u_sum += arr[0] 
        mu = u_sum - arr[0]
        mus.append(mu)
    mus.reverse()
    gen = _gen_sorted_data(lists, desc=False)
    ratio = 0
    for likelihood in likelihoods:
        while likelihood > ratio:
            ratio, arr = next(gen)
            m_sum += arr[1]
        lamb = m_sum - arr[1]
        lambdas.append(lamb)
    return zip(mus, lambdas)

def error_rates_from_threshold(u, m, p, prob, b=None):
    """
    Takes (c,n) shaped arrays I{u} and I{m}, the marginal
    probability of a correct match I{p}, and a threshold
    on the posterior probability of a correct match. Returns
    the error rates corresponding to the threshold.

    If supplied, C{b} is a mapping of variable indices to indices
    of levels to be used for blocking (generally the highest
    similarity score level).

    @type     u: numpy array
    @param    u: (c,n) shaped array of I{u}-probabilities
    @type     m: numpy array
    @param    m: (c,n) shaped array of I{m}-probabilities
    @type     p: C{float}
    @param    p: marginal probability of a correct match
    @type  prob: C{float}
    @param prob: threshold on the posterior
                 probability of a correct match
    @type     b: C{None} or C{dict}
    @param    b: mapping of variable indices to blocking levels
    @rtype:      C{tuple}
    @return:     tuple containing (mu, lambda) for the threshold
    """
    return error_rates_from_thresholds(u, m, p, [prob], b=b)[0]

def thresholds_from_error_rates(u, m, p, mu, lamb, b=None):
    """
    Takes (c,n) shaped arrays I{u} and I{m}, the marginal
    probability of a correct match I{p}, and error rates
    mu and lambda. Returns the parameters for the decision
    rule that provides the desired error rates.
    The parameters for the decision rule are the threshold on the posterior
    probability of a correct match and the probability with which a
    record pair should be allocated to the relevant class if the posterior
    probability is equal to the threshold. These are returned as two pairs,
    the first pair corresponding to the lower threshold and allocation to A3.

    If supplied, C{b} is a mapping of variable indices to indices
    of levels to be used for blocking (generally the highest
    similarity score level).

    @type     u: numpy array
    @param    u: (c,n) shaped array of I{u}-probabilities
    @type     m: numpy array
    @param    m: (c,n) shaped array of I{m}-probabilities
    @type     p: C{float}
    @param    p: marginal probability of a correct match
    @type    mu: C{float}
    @param   mu: the probability of allocation to A1 given an incorrect match
    @type  lamb: C{float}
    @param lamb: the probability of allocation to A3 given a correct match
    @type     b: C{None} or C{dict}
    @param    b: mapping of variable indices to blocking levels
    @rtype:      C{tuple}
    @return:     parameters for lower threshold, and parameters for upper threshold
    @raises:     ValueError if blocking is inconsistent with the lower bound
                 on lambda, or if the calculated thresholds are inconsistent
    """
    lists, m_sum = _get_lists(u, m, b)
    if m_sum > lamb:
        raise ValueError('Blocking implies a lower bound of %s on lambda' % m_sum)
    #
    u_sum = 0
    for u_ratio, arr in _gen_sorted_data(lists, desc=True):
        u_sum += arr[0]
        if u_sum > mu:
            # calculate Pu
            Pu = (mu - (u_sum - arr[0])) / arr[0]
            break
    for m_ratio, arr in _gen_sorted_data(lists, desc=False):
        m_sum += arr[1]
        if m_sum > lamb:
            # calculate Pm
            Pm = (lamb - (m_sum - arr[1])) / arr[1]
            break
    u_prob = u_ratio*p / (1-p + u_ratio*p)
    m_prob = m_ratio*p / (1-p + m_ratio*p)
    if m_prob > u_prob or (m_prob == u_prob and Pm + Pu > 1):
        raise ValueError('Inadmissible values for mu and lambda:\nTry lower error rates')
    return (m_prob, Pm), (u_prob, Pu)

def threshold_from_ratio(u, m, p, ratio, b=None):
    """
    Takes (c,n) shaped arrays I{u} and I{m}, the marginal
    probability of a correct match I{p}, and a ratio for
    lambda / mu. Returns the parameters for the decision
    rule that provides the desired ratio for the error rates and the
    values for mu and lambda under the decision rule.
    The parameters for the decision rule are the threshold on the posterior
    probability of a correct match and the probability with which a
    record pair should be assigned to A1 if the posterior probability is
    equal to the threshold.

    If supplied, C{b} is a mapping of variable indices to indices
    of levels to be used for blocking (generally the highest
    similarity score level).

    @type      u: numpy array
    @param     u: (c,n) shaped array of I{u}-probabilities
    @type      m: numpy array
    @param     m: (c,n) shaped array of I{m}-probabilities
    @type      p: C{float}
    @param     p: marginal probability of a correct match
    @type  ratio: C{float}
    @param ratio: threshold on the posterior
                  probability of a correct match
    @type      b: C{None} or C{dict}
    @param     b: mapping of variable indices to blocking levels
    @rtype:       C{tuple}
    @return:      tuple containing (mu, lambda) for the threshold
    """
    # single iterator approach
    lists, sum_ = _get_lists(u, m, b)
    u_sum = 0
    m_sum = 1
    for u_rat, arr in _gen_sorted_data(lists, desc=True):
        # linear search
        u_sum += arr[0]
        m_sum -= arr[1]
        if m_sum/u_sum < ratio:
            break
    else:
        if sum_/u_sum >= ratio:
            return 0.0, 1.0, u_sum, m_sum
    m_sum = sum_
    gen = _gen_sorted_data(lists, desc=False)
    for rat, arr in gen:
        m_sum += arr[1]
        if rat == u_rat:
            u_sum -=  arr[0]
            break
    while m_sum/u_sum < ratio:
        rat, arr = next(gen)
        u_sum -= arr[0]
        m_sum += arr[1]
    Pu = (m_sum - ratio * u_sum) / (arr[1] + ratio * arr[0])
    mu = u_sum + Pu * arr[0]
    lamb = m_sum - Pu * arr[1]
    m_odds = rat * p/(1-p)
    prob = m_odds/(1+m_odds)
    return prob, Pu, mu, lamb

def threshold_from_cost_function(C_MU, C_UM, C_UU=0, C_MM=0):
    """
    Takes costs of allocation and returns a threshold on
    the posterior proability of a correct match that
    minimises total cost.

    Assumes that C_MU > C_UU and C_UM > C_MM.

    @type  C_MU: C{int} or C{float}
    @param C_MU: cost of allocating to M when true class is U
    @type  C_UM: C{int} or C{float}
    @param C_UM: cost of allocating to U when true class is M
    @type  C_UU: C{int} or C{float}
    @param C_UU: cost of allocating to U when true class is U
    @type  C_MM: C{int} or C{float}
    @param C_MM: cost of allocating to M when true class is M
    @rtype:      C{float}
    @return:     threshold on the prosterior probability of a correct match
                 that minimises the total cost
    """
    return (C_MU - C_UU) / (C_MU - C_UU + C_UM - C_MM)

def threshold_from_linear_function(p, alpha, beta):
    """
    Returns the threshold that minimises the cost function:

    z = alpha*mu + beta*lambda

    @type      p: C{float}
    @param     p: marginal probability of a correct match
    @type  alpha: C{float}
    @param alpha: false negative rate
    @type   beta: C{float}
    @param  beta: false positive rate
    @rtype:       C{tuple}
    @return:      threshold on the posterior probability of a correct match
                  that minimises z
    """
    return alpha*p / (alpha*p + beta*(1-p))

def _get_lists(u, m, b=None):
    # m has shape (c,n)
    # u has shape (c,n)
    # b is a mapping of variable indices to blocking levels
    # generate all m/u
    # generate list of sublists
    b = b or {}
    c, n = u.shape
    lists = []
    m_prod = 1
    for i in range(n):
        if i in b:
            # explicit match
            m_prod *= m[b[i],i]
            lis = [numpy.array([u[j,i], m[j,i]]) for j in [b[i]] if (u[j,i] or m[j,i])]
            if lis:
                lists.append(lis)
        else:
            lis = [numpy.array([u[j,i], m[j,i]]) for j in range(c) if (u[j,i] or m[j,i])]
            if lis:
                lists.append(lis)
    return lists, 1-m_prod


def _gen_sorted_data(lists, desc):
    # desc is True if descending
    # sort lists
    #with numpy.errstate(divide='ignore'):
    lists = [sorted(lis, reverse=desc, key=lambda arr: arr[1]/arr[0]) for lis in lists]
    #print lists
    mult = [1, -1][desc]
    # generate data
    heap = []
    max_index = [len(lis)-1 for lis in lists]
    index = [0]*len(lists)
    arr = _get_vector_data(lists, index)
    #with numpy.errstate(divide='ignore'):
    ratio = mult * arr[1]/arr[0]
    heappush(heap, (ratio, list(arr), index))
    cnt = 0
    last_ratio = -1
    last_arr = numpy.array([0,0])
    while heap:
        cnt += 1
        ratio, arr, index = heappop(heap)
        arr = numpy.array(arr)
        if ratio == last_ratio:
            last_arr += arr
        else:
            if not last_ratio == -1:
                yield mult * last_ratio, last_arr
            last_ratio, last_arr = ratio, arr
        for index in _children(index, max_index):
            arr = _get_vector_data(lists, index)
            #with numpy.errstate(divide='ignore'):
            heappush(heap, (mult * arr[1]/arr[0], list(arr), index))
    if not last_ratio == -1:
        yield mult * last_ratio, last_arr
    #print 'count', cnt
        
def _get_vector_data(lists, index):
    # returns an array for comparison
    # vector, array([u, m])
    prod = 1
    for i, j in enumerate(index):
        prod *= lists[i][j]
    return prod

def _children(node, max_index):
    # node is a list e.g. [0,2,1]
    # lex smallest parent
    for i, val in enumerate(node):
        if val < max_index[i]:
            new_node = list(node)
            new_node[i] += 1
            yield new_node
        if val:
            break


############## plots ##############


def plot_error_rates(filename, u, m, p, b=None, axis=None, format='jpeg', dpi=300, **kwargs):
    """
    Takes (c,n) shaped arrays I{u} and I{m} and the marginal
    probability of a correct match I{p}; then plots the admissible
    region for error rates mu and lambda. The curve corresponds to error
    rates pairs under single thresholds (empty A2). Points below the
    curve correspond to possible error rate pairs with non empty A2.
    Returns the parameters for the decision
    rule that provides the desired error rates.

    If supplied, C{b} is a mapping of variable indices to indices
    of levels to be used for blocking (generally the highest
    similarity score level).

    Any keyword arguments in **kwargs are passed to
    the Matplotlib 'plot' function.

    @type  filename: C{str}
    @param filename: location of saved file
    @type         u: numpy array
    @param        u: (c,n) shaped array of I{u}-probabilities
    @type         m: numpy array
    @param        m: (c,n) shaped array of I{m}-probabilities
    @type         p: C{float}
    @param        p: marginal probability of a correct match
    @type      axis: C{None} or C{list}
    @param     axis: Matplotlib axis
    @type       adj: C{None} or C{dict}
    @param      adj: mapping of variable indices to blocking levels
    """
    lists, m_sum = _get_lists(u, m, b)
    gen = _gen_sorted_data(lists, desc=True)
    x = [0]
    y =[]
    for u_ratio, arr in gen:
        x.append(x[-1]+arr[0])
        y.append(arr[1])
    y.append(m_sum)
    # accumulate the y
    for i in range(-1, -len(y), -1):
        y[i-1] = y[i-1] + y[i]
    # plot it
    import pylab
    pylab.plot(x, y, **kwargs)
    if not axis:
        xmin, xmax = pylab.xlim()
        pylab.xlim(xmin=xmin-0.02*(xmax-xmin))
        ymin, ymax = pylab.ylim()
        pylab.ylim(ymin=ymin-0.02*(ymax-ymin))
    else:
        pylab.axis(axis)
    if m_sum:
        pylab.annotate(r"$\lambda \geq %s$" % m_sum, xy=(0.6,0.9), xycoords='axes fraction')
        pylab.annotate(r"$\mu \leq %s$" % x[-1], xy=(0.6,0.8), xycoords='axes fraction')
    pylab.xlabel(r"$\mu$")
    lab = pylab.ylabel(r"$\lambda$")
    pylab.savefig(filename, format=format, dpi=dpi)
    pylab.clf()
