
import sys, os
import numpy as np
import pycpd

def readObj(path):
    """
    read mesh from wavefront .obj
    :return: (vertices, faces) as numpy arrays
    """
    verts=[]
    faces=[]
    with open(path, 'r') as obj:
	    for ln in obj:
	        if ln.startswith("v "):
	            vert=ln.rstrip().split(' ')[-3:]
	            verts.append([float(v) for v in vert])
	        elif ln.startswith("f "):
	            face=ln.rstrip().split(' ')[-3:]
	            # remove normal id (keep indices 1-based)
	            face=[int(f.split('/')[0]) for f in face]
	            faces.append(face)

    return (verts, faces)

def callback(iteration, error, X, Y):
	print('iteration={0}, error={1}'.format(iteration, error))

def main(argv=sys.argv):
	if len(argv) < 3:
		print('Usage: {0} path_to_obj1 path_to_obj2'.format(argv[0]))
	
	# read objs
	v1, _ = readObj(argv[1])
	X = np.asarray(v1)
	v2, f2 = readObj(argv[2])
	Y = np.asarray(v2)

	# run coherent point drift matching: affine stage
	reg = pycpd.affine_registration(**{ 'X': X, 'Y': Y })
	Y, _ = reg.register(callback)
	
	# run coherent point drift matching: deformable stage
	reg = pycpd.deformable_registration(**{ 'X': X, 'Y': Y })
	Y, _ = reg.register(callback)

	# write obj
	with open(argv[2].replace('.obj', '_to_'+argv[1]), 'w') as out:
		for v in Y:
			out.write('v {0} {1} {2}\n'.format(*v))
		for f in f2:
			out.write('f {0}/{0} {1}/{1} {2}/{2}\n'.format(*f))

if __name__ == '__main__':
	main(sys.argv)