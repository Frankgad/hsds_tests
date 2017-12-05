#pylint: disable=missing-docstring

import time
import argparse
from functools import partial
import sys

import numpy as np
import h5py
import h5pyd
import tables

##################################
## Benchmark tests
##################################
def random_fancy_select(arr, points):
    '''
    Benchmarks fancy indexing (using the : operator)
    on random surface points.
    This is equal to selecting a random trace in a
    3D cube.
    '''
    for point in points:
        x1, y1, z1, x2, y2, z2 = point
        arr[x1, y1, :]

def random_point_select(arr, points):
    '''
    Benchmarks indexing of a random point in 3D.
    This is equal to selecting a random depth in a random trace.
    '''
    for point in points:
        x1, y1, z1, x2, y2, z2 = point
        arr[x1, y1, z1]

def random_cube_select(arr, points):
    '''
    Benchmarks selecting a 3D cube of a random
    size and location.
    '''
    for point in points:
        x1, y1, z1, x2, y2, z2 = np.sort(point)
        arr[x1:x2, y1:y2, z1:z2]

def slice_along_axis(ax, arr, points):
    '''
    Benchmarks slicing a array in
    the first directions (along the x-axis)
    '''
    for point in points:
        x1, y1, z1, z2, x2, y2 = point
        if ax == 0:
            idx = slice(x1, None, None)
        elif ax == 1:
            idx = slice(None, y1, None)
        else:
            idx = slice(None, None, max(z1, z2))

        arr[idx]

##################################
## Helper functions
##################################
def run_test(fin, lib, repeat, func, idx):
    if lib == 'h5py':
        fin = h5py.File(fin, 'r')
        arr = fin['/test']

    elif lib == 'h5pyd':
        fin = h5pyd.File(fin, 'r')
        arr = fin['/test']
		
    else:
        raise IOError("Cound not open file using %s" % lib)

    start = time.time()
    func(arr, idx)
    end = time.time()

    fin.close()
    runtime = 1/((end - start)/repeat)
    print_stats(runtime)
    return runtime

def print_name(name):
    sys.stdout.write("%-40s |" % name)
    sys.stdout.flush()

def print_stats(rtime):
    sys.stdout.write("{0:10.2f} |".format(rtime))
    sys.stdout.flush()


##################################
## Main
##################################
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('h5lib', metavar='l', type=str, nargs='*',
                    choices=['h5py', 'h5pyd'],
                    help='HDF5 library to benchmark')
parser.add_argument('--benchmarks', nargs='?', type=list, default=[],
                    help="Which benchmarks to run")
parser.add_argument('--size', nargs=3, type=int,
                    help='Size of hdf5 test cube (Default: 512)')
parser.add_argument('--write', type=bool, default=0,
                    help='write a hdf5 cube to test (Default: False)')
parser.add_argument('--repeat', type=int, default=500,
                    help='Number of times to repeat benchmarks (Default: 500)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42)')
parser.add_argument('--compression', type=str, default=None,
                    choices=['gzip', 'lzf', 'None'],
                    help='Lossless compression filter (Default: None)')
parser.add_argument('--chunks', type=int, nargs=3,
                    help='Chunck size NxNxN (Default: Auto)')
parser.add_argument('--h5path', type=str, default='/tmp/test.h5',
                    help='Path to HDF5 testfile (Default: /tmp/test.h5')

args = parser.parse_args()

if 'h5py' in args.h5lib:
    print('Benchmarking with h5py')

if 'h5pyd' in args.h5lib:
    print('Benchmarking with h5pyd')

h5path = args.h5path
print()
arr_shape = 0
if args.write == 0:
    if 'h5py' in args.h5lib:
        fin = h5py.File(h5path, 'r')
        ARR = fin['/test']
        shape = ARR.shape
        fin.close()
        N = shape[0]
        x, y, z = shape

    else:
        fin = h5pyd.File(h5path, 'r')
        ARR = fin['/test']
        shape = ARR.shape
        fin.close()
        N = shape[0]
        x, y, z = shape

repeat = args.repeat
np.random.seed(args.seed)

if args.write:
    x, y, z = args.size
    ARR = np.ones((int(x), int(y), int(z))).astype(np.float32)

    if args.chunks:
        chunk = tuple(args.chunks)
    else:
        chunk = True

 #   with:
	if 'h5py' in args.h5lib:
		f = h5py.File(h5path, 'w')
		f.create_dataset("test", dtype='f8',
                        compression=args.compression, chunks=chunk, data=ARR)
	else:
			f = h5pyd.File(h5path, 'w') 
			f.create_dataset("test", dtype='f8', data=ARR)
 	f.flush()
    exit
else:
    random_point = np.array(np.round(np.random.rand(repeat, 6)), dtype=np.int)
    random_point[:, 0] *= (shape[0]-2)
    random_point[:, 3] *= (shape[0]-2)
    random_point[:, 1] *= (shape[1]-2)
    random_point[:, 4] *= (shape[1]-2)
    random_point[:, 2] *= (shape[2]-2)
    random_point[:, 5] *= (shape[2]-2)
    random_point[:, 5] += 1

    runtime = 0
    sys.stdout.write("%-40s |" % 'Benchmarks')
    for r in args.h5lib:
        sys.stdout.write('{:>10s} |'.format(r))

    sys.stdout.write('\n')
    sys.stdout.write('-'*(40+13*len(args.h5lib))+'\n')
    sys.stdout.flush()

    if len(args.benchmarks) == 0:
        print()
        print("Nothing to benchmark")
        exit()

    if '0' in args.benchmarks:
        print_name("Random trace selection")

        for l in args.h5lib:
            runtime = run_test(h5path, l, repeat, partial(random_fancy_select),
                               random_point)

        sys.stdout.write('\n')
        sys.stdout.flush()

    if '4' in args.benchmarks:
        print_name("Random point selection")

        for l in args.h5lib:
            runtime = run_test(h5path, l, repeat, partial(random_point_select),
                               random_point)

        sys.stdout.write('\n')
        sys.stdout.flush()

    if '5' in args.benchmarks:
        print_name("Random subcube selection")

        for l in args.h5lib:
            runtime = run_test(h5path, l, repeat, partial(random_cube_select),
                               random_point)

        sys.stdout.write('\n')
        sys.stdout.flush()

    if '1' in args.benchmarks:
        print_name("Random slicing along x-axis")
        for l in args.h5lib:
            runtime = run_test(h5path, l, repeat, partial(slice_along_axis, 0),
                               random_point)
        sys.stdout.write('\n')
        sys.stdout.flush()

    if '2' in args.benchmarks:
        print_name("Random slicing along y-axis")
        for l in args.h5lib:
            runtime = run_test(h5path, l, repeat, partial(slice_along_axis, 1),
                               random_point)
        sys.stdout.write('\n')
        sys.stdout.flush()


    if '3' in args.benchmarks:
        print_name("Random slicing along z-axis")
        for l in args.h5lib:
            runtime = run_test(h5path, l, repeat, partial(slice_along_axis, 2),
                               random_point)

        sys.stdout.write('\n')
        sys.stdout.flush()

    sys.stdout.write("\n\nUnit: Lookups per second\n")
    sys.stdout.write("Cube size: %ix%ix%i\n" % (shape[0], shape[1], shape[2]))
