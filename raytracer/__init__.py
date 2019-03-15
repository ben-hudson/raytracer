import numpy as np
import copy
import logging
import ezdxf

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# general geometry classes
class Line(object):
    def __init__(self, p0, d):
        super(Line, self).__init__()
        self.p0 = np.array(p0)
        self.d = d / np.linalg.norm(d)

        self.eps = 1e-6 # tolerance

    def __repr__(self):
        return '{}<p={}, d={}>'.format(self.__class__.__name__, self.p0, self.d)

    def intersects(self, seg):
        a = np.dot(self.d, seg.n)
        if np.abs(a) > self.eps:
            b = np.dot(seg.p0 - self.p0, seg.n)
            if np.abs(b) > self.eps:
                d = b/a
                return d
        return None

    def distance(self, point):
        u = point - self.p0
        dt = np.dot(u, self.d) # tangential distance
        # TODO: add normal to line and use this
        dn = np.linalg.norm(u - self.d*dt) # normal distance
        return dt, dn

class LineSegment(Line):
    def __init__(self, p0, p1):
        super(LineSegment, self).__init__(p0, np.array(p1) - np.array(p0))
        self.p1 = np.array(p1)
        self.n = np.array([-self.d[1], self.d[0]]) # ew

    def __repr__(self):
        return '{}<p=[{}, {}], n={}>'.format(self.__class__.__name__, self.p0, self.p1, self.n)

    def contains(self, point):
        dt, dn = self.distance(point)
        return np.abs(dn) < self.eps and dt > 0 and dt <= np.linalg.norm(self.p1 - self.p0)


# raytracing-specific classes
C = 299792458/1e9

class Ray(Line):
    def __init__(self, p0, d, tof=0, speed=C, depth=0):
        super(Ray, self).__init__(p0, d)
        self.tof = tof
        self.speed = speed
        self.depth = depth

    def prop(self, dist):
        self.p0 = self.p0 + self.d*dist
        self.tof += dist/self.speed
        self.depth += 1

    def __repr__(self):
        return '{}<p={}, d={}, t={}>'.format(self.__class__.__name__, self.p0, self.d, self.tof)

class Boundary(LineSegment):
    def __init__(self, p0, p1, n0, n1):
        super(Boundary, self).__init__(p0, p1)
        # refractive indices
        # normal points towards n0
        self.n0 = n0
        self.n1 = n1

    # https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
    def reflect(self, ray):
        if logger.isEnabledFor(logging.WARNING) and not self.contains(ray.p0):
            logger.warn('Reflecting ray that does not originate on boundary')

        # algorithm assumes edge normal is pointing towards ray
        if np.dot(ray.d, self.n) < 0:
            n = self.n
        else:
            n = -self.n
        cosi = np.dot(-ray.d, n)
        d = ray.d + 2*cosi*n
        d = d / np.linalg.norm(d)

        refl = copy.deepcopy(ray)
        refl.d = d
        return refl

    def refract(self, ray):
        # print('refracting')
        if logger.isEnabledFor(logging.WARNING) and not self.contains(ray.p0):
            logger.warn('Refracting ray that does not originate on boundary')

        if np.dot(ray.d, self.n) < 0:
            n = self.n
            n_ratio = self.n0/self.n1
        else:
            n = -self.n
            n_ratio = self.n1/self.n0
        cosi = np.dot(-ray.d, n)
        sinisq = (n_ratio**2)*(1 - cosi**2)

        if sinisq <= 1:
            d = n_ratio*ray.d +(n_ratio*cosi - np.sqrt(1 - sinisq))*n
            d = d / np.linalg.norm(d)

            refr = copy.deepcopy(ray)
            refr.d = d
            refr.speed *= n_ratio
            return refr
        else:
            logger.debug('Total internal reflection')

        return None

class Obstacle(object):
    """docstring for Polygon"""
    def __init__(self, vertices, n_out, n_in):
        super(Obstacle, self).__init__()
        self.vertices = vertices # vertices are in clock-wise order
        self.boundaries = [Boundary(*x, n_out, n_in) for x in zip(vertices, vertices[1:] + vertices[:1])]

    def __repr__(self):
        return '{}<v={}>'.format(self.__class__.__name__, self.vertices)

    @staticmethod
    def from_dxf(filename, ns, scale=1.0):
        drawing = ezdxf.readfile(filename)
        # splines must be exported as line segments
        lines = [(e.dxf.start, e.dxf.end) for e in drawing.query('LINE')]

        while len(lines) > 0:
            obstacle = [lines.pop(0)]
            while True:
                try:
                    # find line where start matches previous end
                    line_i = next(i for i, l in enumerate(lines) if l[0] == obstacle[-1][1])
                    line = lines.pop(line_i)
                    obstacle.append(line)
                except StopIteration:
                    break

            # compute signed area
            # http://mathworld.wolfram.com/PolygonArea.html
            A = 0
            for start, end in obstacle:
                A += start[0]*end[1] - start[1]*end[0]
            if A > 0:
                obstacle.reverse()

            n_out, n_in = ns.pop(0)
            vertices = [(x*scale, y*scale) for (x, y, z), end in obstacle]
            yield Obstacle(vertices, n_out, n_in)

def closest_hit(ray, env):
    hits = []
    for obst in env:
        for b in obst.boundaries:
            d = ray.intersects(b)
            if d and d > 0:
                p = ray.p0 + d*ray.d
                if b.contains(p):
                    hits.append((d, p, b))
    if len(hits) > 0:
        return min(hits, key=lambda x: x[0])

    return None
