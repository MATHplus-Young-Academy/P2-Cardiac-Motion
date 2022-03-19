import pyvista as pv
import numpy as np

from joblib import Parallel, delayed
from joblib import parallel_backend

from morphomatics.manifold import Kendall
from morphomatics.manifold.Bezierfold import Bezierfold
from morphomatics.stats import RiemannianRegression

np.set_printoptions(precision=4)


def compute(degree, n_knees):
    """
    :param degree: degree of Bezier curves
    :param n_knees: use n-knees geodesics
    :param visualize_subject_trends: if True, the individual shape trajectories are visualized
    :return: Gram matrix and mean curve
    """
    # G = pickle.load( open( "gram_matrix_full", "rb" ) )
    # a = scipy.linalg.eig(G)[0]

    # read in shapes (encoded as n-by-2 array)
    surfs = [] # <- surf[i][j] holds j-th frame of i-th subject
    
    # setup shape space and Bézierfold
    M = Kendall(surfs[0][0].shape)
    B = Bezierfold(M, degree)

    # map to shape space
    surfs = [[M.to_coords(s) for s in surfs_i] for surfs_i in surfs]
    # set corresponding times
    times = [np.linspace(0., len(degree), len(surfs_i)+1)[:-1] for surfs_i in surfs] # <- does this make sense?

    def reg(Y, t):
        return RiemannianRegression(M, Y, t, degree, maxiter=1000).trend

    # compute subject-wise trends
    with parallel_backend('multiprocessing'):
        subjecttrends = Parallel(n_jobs=-1, prefer='threads', require='sharedmem', verbose=10)(delayed(reg)(*a) for a in zip(surfs, times))

    # # visualize subject trends
    # for i, gam in enumerate(subjecttrends):
    #     update_mesh = lambda t: p.update_coordinates(M.from_coords(gam.eval(t)))
    #
    #     p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
    #     mesh = pv.PolyData(M.from_coords(gam.eval(0)), pyT.faces)
    #     p.add_text('Trajectory ' + str(i), font_size=24)
    #     p.add_mesh(mesh)
    #     p.reset_camera()
    #     slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1), pointa=(0.4, .85), pointb=(0.9, .85), )
    #     p.show()

    # compute mean trajectory and geodesics to the data curves
    mean, F = B.mean(subjecttrends, n=n_knees, delta=1e-5, min_stepsize=1e-5, nsteps=20, eps=1e-5, n_stepsGeo=5,
                     verbosity=2)

    F_controlPoints = []
    for ff in F:
        s = []
        for f in ff:
            s.append(f.control_points)
        F_controlPoints.append(s)

    # """Visualization"""
    #
    # update_mesh = lambda t: p.update_coordinates(M.from_coords(mean.eval(t)))
    #
    # p = pv.Plotter(shape=(1, 1), window_size=[2548, 2036])
    # mesh = pv.PolyData(M.from_coords(mean.eval(0)), pyT.faces)
    # p.add_text('Mean trajectory for type ' + type, font_size=24)
    # p.add_mesh(mesh)
    # p.reset_camera()
    # slider = p.add_slider_widget(callback=update_mesh, rng=(0, 1), pointa=(0.4, .85), pointb=(0.9, .85), )
    # p.show()

    return mean, F_controlPoints