#! /usr/bin/env python

"""
Script for plotting 2D data or 2D slices of 3D data, intended primarily for
Cartesian grids.

Run "plot_slice.py -h" to see description of inputs.

See documentation on athena_read.athdf() for important notes about reading files
with mesh refinement.

Users are encouraged to make their own versions of this script for improved
results by adjusting figure size, spacings, tick locations, axes labels, etc.
The script must also be modified to plot any functions of the quantities in the
file, including combinations of multiple quantities.
"""

# Python standard modules
import argparse
import warnings
import os

# Other Python modules
import numpy as np

# Athena++ modules
import athena_read

from plot_slice import main
import glob

# Main function
def many_slices(args):

  arg_dict = vars(args)
  format_output = ".png"
  abs_input_path = os.path.abspath(args.data_dir)
  abs_output_path = os.path.abspath(args.output_directory)
  glob_path = os.path.join(abs_input_path, args.file_template)
  input_files = sorted(glob.glob(glob_path))
  
  for i_file in input_files:
    output_file = "{0}{1}".format(os.path.splitext(os.path.split(i_file)[-1])[0], format_output)
    output_path = os.path.join(abs_output_path, output_file)

    arg_dict["data_file"] = i_file
    arg_dict["output_file"] = output_path

    main(**arg_dict)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='name of input directory, possibly including path')
    parser.add_argument('file_template',
                        help='name of file template')
    parser.add_argument('quantity',
                        help='name of quantity to be plotted')
    parser.add_argument('output_directory',
                        help='name of output to be (over)written, possibly including ')
    parser.add_argument('-d', '--direction',
                        type=int,
                        choices=(1, 2, 3),
                        default=3,
                        help=('direction orthogonal to slice for 3D data'))
    parser.add_argument('--slice_location',
                        type=float,
                        default=None,
                        help=('coordinate value along which slice is to be taken '
                              '(default: 0)'))
    parser.add_argument('-a', '--average',
                        action='store_true',
                        help=('flag indicating averaging should be done in orthogonal '
                              'direction for 3D data'))
    parser.add_argument('-s', '--sum',
                        action='store_true',
                        help=('flag indicating summation should be done in orthogonal '
                              'direction for 3D data'))
    parser.add_argument('-l',
                        '--level',
                        type=int,
                        default=None,
                        help=('refinement level to be used in plotting (default: max '
                              'level in file)'))
    parser.add_argument('--x_min',
                        type=float,
                        default=None,
                        help='minimum extent of plot in first plotted direction')
    parser.add_argument('--x_max',
                        type=float,
                        default=None,
                        help='maximum extent of plot in first plotted direction')
    parser.add_argument('--y_min',
                        type=float,
                        default=None,
                        help='minimum extent of plot in second plotted direction')
    parser.add_argument('--y_max',
                        type=float,
                        default=None,
                        help='maximum extent of plot in second plotted direction')
    parser.add_argument('-f', '--fill',
                        action='store_true',
                        help='flag indicating image should fill plot area, even if this '
                             'distorts the aspect ratio')
    parser.add_argument('-c',
                        '--colormap',
                        default=None,
                        help=('name of Matplotlib colormap to use instead of default'))
    parser.add_argument('--vmin',
                        type=float,
                        default=None,
                        help=('data value to correspond to colormap minimum; use '
                              '--vmin=<val> if <val> has negative sign'))
    parser.add_argument('--vmax',
                        type=float,
                        default=None,
                        help=('data value to correspond to colormap maximum; use '
                              '--vmax=<val> if <val> has negative sign'))
    parser.add_argument('--logc',
                        action='store_true',
                        help='flag indicating data should be colormapped logarithmically')
    parser.add_argument('--stream',
                        default=None,
                        help='name of vector quantity to use to make stream plot')
    parser.add_argument('--stream_average',
                        action='store_true',
                        help='flag indicating stream plot should be averaged in '
                             'orthogonal direction for 3D data')
    parser.add_argument('--stream_density',
                        type=float,
                        default=1.0,
                        help='density of stream lines')
    parser.add_argument('--num_ghost',
                        type=int,
                        default=0,
                        help=('Include number of ghost cells in each direction'))
    args = parser.parse_args()
    many_slices(args)
