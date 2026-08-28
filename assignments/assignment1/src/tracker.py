from typing import List, Union

import numpy as np

from .utils import sample_normal
from data_types.nominal_types import vct3, Frame, Rot, vct3Array
from data_types.uncertain_types import uvct3, uFrame
from data_types.covariance_types import Covariance
from .marker import MarkerBody
from uncertainty_networks.se3 import exp_se3


class Tracker:

    def __init__(self, name, F: Union[Frame, uFrame]):
        self.name = name
        self.F = F  # Frame position of the tracking camera in the world coordinate system; set by students
        self.marker_positions = []  # observed marker positions in the tracker's coordinate frame

    def __repr__(self):
        return f"Tracker(name={self.name!r}, F={self.F!r})"

    def set_pose(self, frame: Union[Frame, uFrame]):
        self.F = uFrame(frame)
        return self

    def set_cov(self, cov: Covariance):  # here we might want to pass just np.ndarray, but then we avoid any checks
        self.F = uFrame(self.F, cov)
        return self

    def read_marker_body(self, marker_body: MarkerBody, jiggle_cov: Covariance = None,
                          sensor_noise_cov: Covariance = None, world_frame: Frame = None) -> List[uvct3]: 
        '''
        Reads marker_body's markers through this tracker, in the tracker's
        own coordinate frame.

        input:
            marker_body - a MarkerBody with a cached manufacturing-noise
                ground truth (see MarkerBody.get_actual_marker_positions)
            jiggle_cov - 6x6 matrix: covariance of this tracker's
                OWN pose perturbation ([alpha; epsilon]). One fresh sample is
                drawn per call and shared across every marker read in that
                call (F*_tracker = F_tracker . Exp(eta)).
            sensor_noise_cov - 3x3 matrix: covariance of the
                camera's per-marker detection noise (sigma_b). An
                independent fresh sample is drawn for each marker, every
                call.
            body_placement - marker_body's current actual pose, expressed in
                whatever frame self.F itself is measured against (e.g. the
                workspace). Composed in before applying this tracker's own
                pose, so bodies at different physical locations/orientations
                correctly register to different Frames. Defaults to None
                (marker_body's local frame is treated as already being in
                that same reference frame -- i.e. no separate placement).
        output: samples - list of observed marker positions (List[uvct3]),
            in the tracker's frame
        '''
        F_nom = self.F.F if isinstance(self.F, uFrame) else self.F
        F_actual = F_nom
        if jiggle_cov is not None:
            eta = sample_normal(np.zeros(6), cov=jiggle_cov)[0]  # [alpha(3); epsilon(3)], one draw for this call
            dF = Frame(exp_se3(eta))  # Monte-Carlo perturbation helper: T_true = T_nom . Exp(eta)
            F_actual = F_nom * dF     # F*_tracker = F_tracker . delta_F_tracker

        observed = []
        for actual_pos in marker_body.get_actual_marker_positions():
            local_p = actual_pos.p
            placed_p = (body_placement * local_p) if body_placement is not None else local_p
            p = F_actual.inv() * placed_p  # plain Frame * vct3 -> vct3
            if sensor_noise_cov is not None:
                p = sample_normal(p, cov=sensor_noise_cov)  # independent fresh draw per marker
            observed.append(uvct3(p))

        self.marker_positions = observed
        return observed

    def procrustes_solver(self, nominal_points: List[vct3], observed_points: List[uvct3]) -> Frame:
        '''
        Rigid point-set registration via Arun's SVD method: finds the best-fit
        Frame F (in the least-squares sense) such that
        F * nominal_points[i] ~= observed_points[i], for corresponding pairs.

        input:
            nominal_points - marker body's best-known geometry, in its own
                local coordinate frame
            observed_points - this tracker's paired observations of those
                same markers (e.g. from read_marker_body), in the tracker's
                frame, same order/correspondence as nominal_points
        output: the best-fit Frame mapping nominal_points -> observed_points
        '''
        if len(nominal_points) != len(observed_points) or len(nominal_points) < 3:
            raise ValueError("procrustes_solver requires >= 3 paired points")

        P = vct3Array(nominal_points)
        Q = vct3Array([o.p for o in observed_points])
        p_bar, q_bar = P.mean(), Q.mean()
        Pc, Qc = (P - p_bar).mat, (Q - q_bar).mat

        U, _, Vt = np.linalg.svd(Pc @ Qc.T)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R_mat = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        R = Rot(matrix=R_mat)
        p_vec = q_bar.vec - R_mat @ p_bar.vec
        p = vct3(p_vec[0, 0], p_vec[1, 0], p_vec[2, 0])
        return Frame(R, p)
