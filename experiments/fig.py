#-*- coding:utf-8 -*-
# AUTHOR:
# FILE:     fig.py
# ROLE:     TODO (some explanation)
# CREATED:  2015-05-16 23:37:47
# MODIFIED: 2015-05-16 23:37:47

import matplotlib
import matplotlib.pyplot  as plt
from matplotlib import rc
import logging

matplotlib.use('Agg')
#rc('text', usetex=True)
#rc('font', family='Times New Roman')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True


def plot(xs, labels, ys, x_label, y_label, output_path,
        line_width=3,
        marker_size=12,
        axis=None, xtickers=None, draw_legend=True):

    color_list = [(0.12,0.56,1),(0.58,0.66,0.2),(0.48,0.41,0.93),
            (0,0.75,0.75), 'b','m','k',(0,0.93,0),'r']
    marker_list = ['s','^','v','<','>','h','*',u'o','d']
    line_styles=['-','--']

    c_ind = 0
    m_ind = 0
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    lines=[]
    ind = 0
    line_num = len(xs)
    for i in xrange(len(xs)):
        line, = ax.plot(xs[i],ys[i],
                color=color_list[c_ind % len(color_list)],
                marker=marker_list[m_ind % len(marker_list)],
                linestyle=line_styles[c_ind % len(line_styles)],
                clip_on=True,markersize=marker_size,
                linewidth=line_width,
                #fillstyle='full',
                zorder=100
                )
        c_ind += 1
        m_ind += 1
        lines.append(line)

    #if xtickers == None:
    #    ax.set_xticks(xs[i])
    #else:
    #    ax.set_xticks(xtickers)

    if axis != None:
        ax.axis(axis)
    ax.grid()
    if draw_legend:
        ax.legend(lines,labels,loc='best',ncol=2)
    plt.xlabel(x_label,fontsize=24)
    plt.ylabel(y_label,fontsize=24)
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.savefig(output_path,bbox_inches='tight')
    logging.info('figure saved to %s' %(output_path))
    #plt.show()
