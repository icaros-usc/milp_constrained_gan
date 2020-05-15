#!/usr/category/env python
import numpy as np
from ss_plotting.make_plots import plot_bar_graph

# Pretty version of this plot: http://matplotlib.org/examples/api/barchart_demo.html

categories = ['u20', '20-30', '30-40', '40-50', '50+']
men_means = [20, 35, 30, 35, 27]
men_errs = [2, 3, 4, 1, 2]
women_means = [25, 32, 34, 20, 25]
women_errs = [3, 5, 2, 3, 3]

series = [men_means, women_means]
series_labels = ['Men', 'Women']
series_colors = ['green', 'purple']


ylabel = 'Scores'
title = 'Scores by gender and age group'

plot_bar_graph(series, series_colors,
               series_labels=series_labels,
               series_errs = [men_errs, women_errs],
               category_labels = categories,
               plot_ylabel = ylabel,
               plot_title = title)