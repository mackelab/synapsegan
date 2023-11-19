import torch
import numpy as np
import matplotlib.pyplot as plt


def Oja(w, x, y, rule=None):
    dy = x*w - y
    if rule is None:
        dw = y * (x - y*w)
    else:
        dw = rule(w, x, y)
    return [dw, dy]

def rule_wrapper(rule):
    def fwd_pass(w, x, y):
        if type(rule) == torch.nn.Sequential:
            inputs = torch.cat([torch.FloatTensor([w]).reshape(-1, 1), 
                                torch.FloatTensor([x]).reshape(-1, 1), 
                                torch.FloatTensor([y]).reshape(-1, 1)],
                               -1)
            return rule(inputs).data.numpy().squeeze()
        else:
            w = torch.FloatTensor([w]).reshape(1, 1, 1).repeat(1, 3, 1)
            x = torch.FloatTensor([x]).reshape(1, 1).repeat(1, 3)
            y = torch.FloatTensor([y]).reshape(1, 1)
            return rule(w, x, y).data.numpy().squeeze()[0]
    return fwd_pass

def generate_dataset(function, x_range, y_range, w, n_points):
    xs = np.linspace(x_range[0], x_range[1], num = n_points[0])
    ys = np.linspace(y_range[0], y_range[1], num = n_points[1])
    return(xs, ys, np.array([[function(w, x, y) for x in xs] for y in ys]))


def generate_quiver(rule,
                    w_range=np.linspace(-2, 2, 100),
                    y_range=np.linspace(-2, 2, 100),
                    x=0.5):

    W, Y = np.meshgrid(w_range[::5], y_range[::5])
    DW, DY = np.zeros_like(W), np.zeros_like(Y)
    for i, (w, y) in enumerate(zip(W, Y)):
        for j, (ww, yy) in enumerate(zip(w, y)):
            dw, dy = Oja(ww, x, yy, rule)
            denom = 1
            denom = np.sqrt(dw ** 2 + dy**2)
            DW[i, j], DY[i, j] = dw/denom, dy / denom
    return Y, W, DY, DW

def plot_2Dheatmap(function,
                   ax,
                   w,
                   cax = None,
                   x_range = [-1,1],
                   y_range = [-1,1],
                   n_points = [100,100],
                   fontsize = 17,
                   linewidth = 1,
                   font = 'Arial',
                   alpha = 1,
                   labelpad=95,
                   labelsize=20):
    xs, ys, dws = generate_dataset(function, x_range, y_range, w, n_points)

#     dw_min, dw_max = dws.min(), dws.max()
    dw_min, dw_max = -4, 4

    c = ax.pcolormesh(xs, ys, dws,
                      cmap='RdBu', vmin=dw_min, vmax=dw_max)

    ax.set_xlabel(r'$x$', fontsize=labelsize, fontname=font, labelpad = labelpad)
    ax.set_ylabel(r'$y$', fontsize=labelsize, fontname=font, labelpad = labelpad)
    ax.set_aspect(1)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    ax.set_xticks([xs[0], xs[-1]])
    ax.set_xticklabels([int(xs[0]), int(xs[-1])])
    ax.set_yticks([ys[0], ys[-1]])
    ax.set_yticklabels([int(ys[0]), int(ys[-1])])
    ax.set_ylim(ys[0]-0.01, ys[-1]+0.01)
#     ax.set_xlim(xs[0]-0.15, xs[-1]+0.15)

    if cax is not None:
        cbar = plt.colorbar(c, cax=cax, ticks=[dw_min, dw_max], drawedges = False, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize, width = 0)#, position='left')
        cbar.set_label(r'$\Delta \omega$,'+ "  " +r'$\omega = $%.2f' %w,
                       size=20, labelpad = -10)
#         cax.yaxis.set_ticks_position('left')
#         cax.yaxis.set_label_position('left')
    return c


def insets(ax, t, key, panel_test_dataset, colors, labels, test_outputs, ylabel=None, fontsize=17, labelsize=20, titlesize=20,
           labelpad=-2., gt_key="gt", alphas=None):
    ax.set_xlim(-3, 3)
    ax.set_xticks([-2.0, 0.0, 2.0])
    ax.set_ylim(-3, 3)
    ax.set_yticks([-2, 0, 2])
    ax.set_xticklabels(labels=[-2.0, 0.0, 2.0], fontsize=fontsize)
    ax.set_yticklabels(labels=[-2.0, 0.0, 2.0], fontsize=fontsize)
    ax.set_aspect(1)
#

    if t<29:
        if ylabel is None:
            ylabel = labels[key]
        ax.set_ylabel(ylabel, fontsize=labelsize, labelpad=labelpad)
    else:
        ax.set_yticks([])

    ax.set_xlabel("Oja's Rule", fontsize=labelsize)

    ax.set_title("t = %d" % (t+1), loc="center", y=.9, fontsize=titlesize)

    out_gt = test_outputs[gt_key][panel_test_dataset, :, t].reshape(-1)
    ax.plot(out_gt, out_gt,"k", lw=1)
    out = test_outputs[key][panel_test_dataset, :, t].reshape(-1)
    
    alpha = alphas[key] if alphas is not None else 0.6
    ax.scatter(out_gt,
               out,
               color=colors[key],
               alpha=alpha,
               s=120)
