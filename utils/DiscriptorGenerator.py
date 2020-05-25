import numpy as np

class DiscriptorGenerator:
    def __init__(self, trj:np.ndarray, idx:int, cutoff_radius:float=1.0):
        self.idx_a, self.idx_b = self._choose_nearest_2indexes(trj[0], idx)

        self.cutoff_radius = cutoff_radius
    

    def __call__(self, struct:np.ndarray, i:int, force:np.ndarray):
        # shift
        Ri = struct - struct[i]

        # rotate
        e1 = self._e(Ri[self.idx_a])
        e2 = self._e(Ri[self.idx_b] - np.dot(Ri[self.idx_b],e1)*e1)
        e3 = np.cross(e1, e2)
        rotation_martix = np.array([e1,e2,e3]).T
        Ri = np.dot(Ri, rotation_martix)

        Ri = np.delete(Ri, obj=i, axis=0)

        radius_list = [self.radius(Ri[j]) for j in range(Ri.shape[0])]

        # descriptor
        D = np.array([ \
            [1/radius_list[j], Ri[j,0]/radius_list[j], Ri[j,1]/radius_list[j], Ri[j,2]/radius_list[j]] \
            for j in range(Ri.shape[0]) \
            if radius_list[j]<=self.cutoff_radius \
            ])
        
        # sort
        D = np.sort(D, axis=0)[::-1]
        
        # also rotate force
        force = np.dot(force, rotation_martix)
        
        return D, force



    def radius(self, a:np.ndarray, b=np.array([0,0,0])) -> float:
        return np.sqrt(np.sum(np.square(a-b)))


    def _choose_nearest_2indexes(self, struct:np.ndarray, i:int) -> np.ndarray:
        radiuslist = [self.radius(struct[i], struct[j]) for j in range(struct.shape[0])]
        return np.argsort(radiuslist)[1:3]


    def _e(self, R:np.ndarray) -> np.ndarray:
        return R/self.radius(R)
