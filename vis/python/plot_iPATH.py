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


import matplotlib.pyplot as plt   
import matplotlib.colors as colors

import glob

# root_dir = '../../../data2/'
root_dir = '/media/store/krs/build/athena/test/athena_cyl'
# root_dir = '/media/store/krs/build/athena/test_junxiang'
n_start = 0
n_end   = 50

x1dims = 400
x1min = 0.1
x1max = 2.0
x2max = 360.0
x2dims = 240
max_n = 50.0


# create grids
x1a = [0.0]*x1dims

for i in range(0, x1dims):
       x1a[i] = x1min+ i * (x1max-x1min)/(x1dims-1)

x3a = []
for i in range(0, x2dims):
       x3a.append(float(i)*x2max/float(x2dims-1))
# print(len(x3a), x3a)
# quit()
azimuths = np.radians(x3a)

r, theta = np.meshgrid(x1a, azimuths)

# for field lines:
num = 8  # number of field lines
theta_space = float(x2max)/num
dx1a = x1a[2] - x1a[1]
dx3a = x3a[2] - x3a[1]
dt = 0.0002
maxt = 20000
pi = 3.14159265359

# Athena variable
level = None  
#level = 'level' #refinement level
ngh = 2  # number of ghost cells. default is 2


# for each HDF file:
# for ii in range(n_end, n_start-1, -1):
for ii in range(n_end+1):
       file_no = os.path.join(root_dir,"solar_wind.out3.00{:03d}.athdf".format(ii))

       data = athena_read.athdf(file_no, quantities=['vel1', 'Bcc1', 'Bcc2', 'rho'], 
              level=level, num_ghost=ngh)

       v1 = data['vel1']
       b1 = data['Bcc1']
       b3 = data['Bcc2']
       dens = data['rho']
       # b1 = b1_data
       # b3 = b3_data
       # dens = dens_data

#       print(data)
       print('dens dimension:', np.shape(dens), 'v1 dimension:', np.shape(v1),
              'b1 dimension:', np.shape(b1) )
       
       
       dens_norm = np.ndarray((x2dims, x1dims))

       for i in range(0, x2dims):
              for j in range(0,x1dims):
                     dens_norm[i,j] = dens[0,i+ngh,j+ngh]*x1a[j]**2.
       # plot field lines

       fl0_r = []
       fl0_th = []
       for i in range(0,num):
              mfl_r = []
              mfl_th = []
              temp_r = x1min
              temp_th = (1.0 + i*theta_space)
       
              while temp_r <= x1max:
                     r_index = (temp_r-x1min)/dx1a
                     th_index = (temp_th)/dx3a

                     # print temp_th , r_index
                     # print b1[int(round(temp_th))-1, 0, int(r_index)], b1[int(round(temp_th)), 1, int(r_index)]
                     # pause
                     dr  = b1[0, int(round(th_index))-1+ngh, int(r_index)+ngh] * dt
                     dth = b3[0, int(round(th_index))-1+ngh, int(r_index)+ngh] * dt / temp_r *180./pi

                     temp_r = temp_r + dr
                     temp_th = temp_th + dth
                     
                     if temp_th < 0:
                            temp_th += 360.
                            if temp_th < 0:
                                   temp_th += 360.
                     if temp_th >= 360:
                            temp_th -= 360.                     
                            if temp_th >= 360:
                                   temp_th -= 360.


                     mfl_r.append(temp_r)
                     mfl_th.append(temp_th/180.*pi)
              fl0_r.append(mfl_r)
              fl0_th.append(mfl_th)


       all_fl_r =[]
       all_fl_th =[]

       mid_pos = [[0.2, 100], [0.5, 100],[1., 100],[1, 100]]
       target = []
       for i in range(0,4):
              mfl_r = []
              mfl_th = []

              temp_r = mid_pos[i][0]
              temp_th = mid_pos[i][1]
              target.append([temp_th/180.*pi, temp_r])

              while temp_r >= x1min:
                     mfl_r.append(temp_r)
                     mfl_th.append(temp_th/180.*pi)

                     r_index = (temp_r-x1min)/dx1a
                     th_index = (temp_th)/dx3a

                     dr  = b1[0, int(round(th_index))-1+ngh, int(r_index)+ngh] * dt
                     dth = b3[0, int(round(th_index))-1+ngh, int(r_index)+ngh] * dt / temp_r *180./pi

                     temp_r = temp_r - dr
                     temp_th = temp_th - dth

                     if temp_th < 0:
                            temp_th += 360.
                     if temp_th >= 360:
                            temp_th -= 360.

              mfl_r.reverse()
              mfl_th.reverse()

              temp_r = mid_pos[i][0]
              temp_th = mid_pos[i][1]

              while temp_r <= x1max:

                     r_index = (temp_r-x1min)/dx1a
                     th_index = (temp_th)/dx3a

                     dr  = b1[0, int(round(th_index))-1+ngh, int(r_index)+ngh] * dt
                     dth = b3[0, int(round(th_index))-1+ngh, int(r_index)+ngh] * dt / temp_r *180./pi

                     temp_r = temp_r + dr
                     temp_th = temp_th + dth
                     
                     if temp_th < 0:
                            temp_th += 360.
                     if temp_th >= 360:
                            temp_th -= 360.

                     mfl_r.append(temp_r)
                     mfl_th.append(temp_th/180.*pi)

              all_fl_r.append(mfl_r)
              all_fl_th.append(mfl_th)

       ticks = []
       

       
       for i in range(0,11):
              ticks.append(i*max_n/10.)

       print (target)

       print (np.min(dens_norm), np.max(dens_norm))
       #fig = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(12,12))
       


       #============= PLOTTING ================================================================
       fig = plt.figure(1, figsize=(8,9))
       ax = fig.add_axes([0.1, 0.03, 0.8, 0.8], projection='polar')
       ax.set_rlabel_position(10)
       ax.set_rticks([0.5, 1.0, 1.5, 2.0])
       ax.set_rorigin(-x1min)
       pcm = ax.contourf(theta, r, dens_norm,  extend='max', fontsize=25,\
              levels = np.linspace(0,max_n,150), cmap ='coolwarm', vmin=0.0, vmax=max_n, yunits ='AU')


       # 1AU circle
       ax.plot(np.linspace(0,2*pi,x2dims), [1.0]*x2dims, 'k', linewidth=2.5)

       for i in range(0,num):
              ax.plot(fl0_th[i], fl0_r[i], 'k')

#       ax.plot(all_fl_th[0], all_fl_r[0], 'w-', linewidth=2.0)
#       ax.plot(target[0][0], target[0][1], 'wo',linewidth=6.0, markersize= 10)

#       for i in xrange(0,3):
#              ax.plot(all_fl_th[i], all_fl_r[i], 'w-', linewidth=1.5)
#              ax.plot(all_fl_th[i], all_fl_r[i], 'k--', linewidth=1.5)
       # for i in range(0,4):
       #        ax.plot(target[i][0], target[i][1], 'wo',linewidth=2.0)

#       ax.plot(70./180.*pi, 1.5, 'ro',linewidth=2.0)

#       ax.annotate('Mars', xy=(mid_pos[0][1]/180.*pi, 1.6), color='w', fontsize=32)
#       ax.annotate('STA', xy=(132./180.*pi, 1.3), color='w', fontsize=32)
#       ax.annotate('STB', xy=(mid_pos[2][1]/180.*pi, 1.3), color='w', fontsize=32)
#       ax.annotate('D', xy=(70/180.*pi, 1.6), color='w', fontsize=32)


#       ax.annotate('E', xy=(mid_pos[0][1]/180.*pi, 1.1), color='r', fontsize=28)
#       ax.annotate('A', xy=(mid_pos[1][1]/180.*pi, 1.1), color='r', fontsize=28)
#       ax.annotate('B', xy=(mid_pos[2][1]/180.*pi, 1.1), color='r', fontsize=28)

            # ,  # theta, radius
            # xytext=(0.05, 0.05),    # fraction, fraction
            # textcoords='figure fraction',
            # arrowprops=dict(facecolor='black', shrink=0.05),
            # horizontalalignment='left',
            # verticalalignment='bottom',
            # )

       ax.set_rlim([x1min, x1max])
       ax.set_rmin(x1min)
       ax.set_rmax(x1max)
       ax.tick_params(axis='both', labelsize=20)



       # magnetic field lines
       cbaxes = fig.add_axes([0.1, 0.9, 0.8, 0.03]) 
       cb = plt.colorbar(pcm, cax = cbaxes,orientation='horizontal',\
             ticks= ticks) 
       cbaxes.tick_params(labelsize=18)
       cbaxes.set_title('$R^2 N(AU^2cm^{-3})$', fontsize=25)


       #cbar=fig.colorbar(pcm, orientation='horizontal', shrink=0.8, aspect=25,\
       #      ticks= ticks, pad=0.1, fraction=0.03 )
       #cbar.ax.set_title('$R^2 N(AU^2cm^{-3})$', fontsize=18)

       #plt.show()
       # break
       plt.savefig(os.path.join(root_dir, "output",'CME{:03d}.png'.format(ii)))


       plt.close(fig)

# # Main function
# def many_slices(args):

#   arg_dict = vars(args)
#   format_output = ".png"
#   abs_input_path = os.path.abspath(args.data_dir)
#   abs_output_path = os.path.abspath(args.output_directory)
#   glob_path = os.path.join(abs_input_path, args.file_template)
#   input_files = sorted(glob.glob(glob_path))
  
#   for i_file in input_files:
#     output_file = "{0}{1}".format(os.path.splitext(os.path.split(i_file)[-1])[0], format_output)
#     output_path = os.path.join(abs_output_path, output_file)

#     arg_dict["data_file"] = i_file
#     arg_dict["output_file"] = output_path

#     main(**arg_dict)


# # Execute main function
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('data_dir',
#                         help='name of input directory, possibly including path')
#     parser.add_argument('output_directory',
#                         help='name of output to be (over)written, possibly including ')
#     parser.add_argument('-d', '--direction',
#                         type=int,
#                         choices=(1, 2, 3),
#                         default=3,
#                         help=('direction orthogonal to slice for 3D data'))
#     parser.add_argument('--slice_location',
#                         type=float,
#                         default=None,
#                         help=('coordinate value along which slice is to be taken '
#                               '(default: 0)'))
#     parser.add_argument('-a', '--average',
#                         action='store_true',
#                         help=('flag indicating averaging should be done in orthogonal '
#                               'direction for 3D data'))
#     parser.add_argument('-s', '--sum',
#                         action='store_true',
#                         help=('flag indicating summation should be done in orthogonal '
#                               'direction for 3D data'))
#     parser.add_argument('-l',
#                         '--level',
#                         type=int,
#                         default=None,
#                         help=('refinement level to be used in plotting (default: max '
#                               'level in file)'))
#     parser.add_argument('--x_min',
#                         type=float,
#                         default=None,
#                         help='minimum extent of plot in first plotted direction')
#     parser.add_argument('--x_max',
#                         type=float,
#                         default=None,
#                         help='maximum extent of plot in first plotted direction')
#     parser.add_argument('--y_min',
#                         type=float,
#                         default=None,
#                         help='minimum extent of plot in second plotted direction')
#     parser.add_argument('--y_max',
#                         type=float,
#                         default=None,
#                         help='maximum extent of plot in second plotted direction')
#     parser.add_argument('-f', '--fill',
#                         action='store_true',
#                         help='flag indicating image should fill plot area, even if this '
#                              'distorts the aspect ratio')
#     parser.add_argument('-c',
#                         '--colormap',
#                         default=None,
#                         help=('name of Matplotlib colormap to use instead of default'))
#     parser.add_argument('--vmin',
#                         type=float,
#                         default=None,
#                         help=('data value to correspond to colormap minimum; use '
#                               '--vmin=<val> if <val> has negative sign'))
#     parser.add_argument('--vmax',
#                         type=float,
#                         default=None,
#                         help=('data value to correspond to colormap maximum; use '
#                               '--vmax=<val> if <val> has negative sign'))
#     parser.add_argument('--logc',
#                         action='store_true',
#                         help='flag indicating data should be colormapped logarithmically')
#     parser.add_argument('--stream',
#                         default=None,
#                         help='name of vector quantity to use to make stream plot')
#     parser.add_argument('--stream_average',
#                         action='store_true',
#                         help='flag indicating stream plot should be averaged in '
#                              'orthogonal direction for 3D data')
#     parser.add_argument('--stream_density',
#                         type=float,
#                         default=1.0,
#                         help='density of stream lines')
#     args = parser.parse_args()
#     main(**vars(args))
