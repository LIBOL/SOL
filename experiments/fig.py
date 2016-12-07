#-*- coding:utf-8 -*-
# AUTHOR:
# FILE:     fig.py
# ROLE:     TODO (some explanation)
# CREATED:  2015-05-16 23:37:47
# MODIFIED: 2015-05-16 23:37:47

import matplotlib.pyplot  as plt
import logging

plt.rc('pdf', fonttype=42)


def plot(xs, ys,
         x_label,
         y_label,
         legends,
         output_path,
         line_width=3,
         marker_size=12,
         xlim=None,
         ylim=None,
         xtickers=None,
         logx=False,
         logy=False,
         clip_on=False,
         legend_cols=2,
         legend_order=201,
         legend_loc='best',
         bbox_to_anchor=None,
         draw_legend=True):

    color_list = ['r', 'm', 'k', 'b', (0.12, 0.56, 1), (0.58, 0.66, 0.2), (0.48, 0.41, 0.93),
            (0, 0.75, 0.75), (0,0.93,0)]
    marker_list = ['s', 'h', '*', u'o', 'd', '^', 'v', '<', '>']
    #line_styles=['-','--']

    c_ind = 0
    m_ind = 0
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lines = []
    if logx is True and logy is True:
        plot_handler = ax.loglog
    elif logx is True and logy is False:
        plot_handler = ax.semilogx
    elif logx is False and logy is True:
        plot_handler = ax.semilogy
    else:
        plot_handler = ax.plot

    for i in xrange(len(xs)):
        zorder = 200 - i
        color = color_list[c_ind % len(color_list)]
        marker = marker_list[m_ind % len(marker_list)]
        c_ind += 1
        m_ind += 1
        if xlim != None:
            x_values = []
            y_values = []
            for k in xrange(len(xs[i])):
                if xs[i][k] >= xlim[0] and xs[i][k] <= xlim[1]:
                    x_values.append(xs[i][k])
                    y_values.append(ys[i][k])
        else:
            x_values = xs[i]
            y_values = ys[i]
        line, = plot_handler(x_values, y_values,
                             color=color,
                             marker=marker,
                             linestyle='-',
                             clip_on=clip_on,
                             markersize=marker_size,
                             linewidth=line_width,
                             fillstyle='full',
                             zorder=zorder)
        lines.append(line)

    if xtickers != None:
        ax.set_xticks(xtickers)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)

    ax.grid()
    if draw_legend:
      l = ax.legend(lines,legends,loc=legend_loc,ncol=legend_cols)
      l.set_zorder(legend_order)
      if bbox_to_anchor != None:
        l.set_bbox_to_anchor(bbox_to_anchor)

    plt.xlabel(x_label,fontsize=18)
    plt.ylabel(y_label,fontsize=18)
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.savefig(output_path,bbox_inches='tight')
    logging.info('figure saved to %s' %(output_path))
    #plt.show()
