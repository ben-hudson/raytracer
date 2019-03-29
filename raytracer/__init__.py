import numpy as np
import ezdxf

C = 299792458/1e9

class Ray(object):
    """docstring for Ray"""
    def __init__(self, origin, direction, speed=C):
        super(Ray, self).__init__()
        self.origin = np.array(origin)
        self.direction = direction / np.linalg.norm(direction)

        self.tof = 0 # time of flight in ns
        self.speed = speed # speed in m/ns

    def distance_to_points(self, points):
        u = points - self.origin
        d_tangential = np.dot(u, self.direction)
        v = u - self.direction*d_tangential[:, np.newaxis]
        d_normal = np.linalg.norm(v, axis=1)
        return np.vstack((d_tangential, d_normal)).T

    def distance_to_segments(self, segments):
        # distance from point to line segments along direction
        # nan if line does not intersect with segment
        # segment directions
        segments_d = segments[:, 1, :] - segments[:, 0, :]
        segments_len = np.linalg.norm(segments_d, axis=1)
        segments_d = segments_d / segments_len[:, np.newaxis]
        # calculate intersections
        # 2x2 matrix inverse might be faster than doing one big one
        # https://math.stackexchange.com/questions/406864/intersection-of-two-lines-in-vector-form
        distances = np.nan*np.ones(segments_d.shape[0])
        for i in range(segments_d.shape[0]):
            A = np.vstack((self.direction, -segments_d[i, :])).T
            b = segments[i, 0, :] - self.origin
            if np.abs(np.linalg.det(A)) > 1e-10:
                x = np.linalg.solve(A, b) # distance along line and segment
                # line intersects with segment if:
                # distance along segment is > 0
                # and distance along segment is < segment length
                # and distance along line is > 0
                if x[1] > 0 and x[1] <= segments_len[i] and x[0] > 1e-10: # this tolerance???
                    distances[i] = x[0]

        return distances

    def closest_hit(self, environment):
        hits = self.distance_to_segments(environment.obstacles)
        if np.isnan(hits).all():
            return None, None
        else:
            i = np.nanargmin(hits)
            return hits[i], environment[i]

    def propagate(self, distance):
        self.origin = self.origin + distance*self.direction
        self.tof += distance/self.speed

    def reflect(self, boundary):
        # https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
        # direction and normal are pointing in towards eachother
        normal = boundary.normal if np.dot(self.direction, boundary.normal) < 0 else -boundary.normal
        cosi = np.dot(-self.direction, normal)
        reflection = self.direction + 2*cosi*normal
        self.direction = reflection / np.linalg.norm(reflection)
        return True # success

    def refract(self, boundary):
        # https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
        # direction and normal are pointing in the same direction
        if np.dot(self.direction, boundary.normal) < 0:
            normal = boundary.normal
            n_ratio = boundary.n[0]/boundary.n[1]
        else:
            normal = -boundary.normal
            n_ratio = boundary.n[1]/boundary.n[0]
        cosi = np.dot(-self.direction, normal)
        sinisq = (n_ratio**2)*(1 - cosi**2)

        if sinisq <= 1:
            refraction = n_ratio*self.direction + (n_ratio*cosi - np.sqrt(1 - sinisq))*normal
            self.direction = refraction / np.linalg.norm(refraction)
            self.speed *= n_ratio
            return True
        else:
            # total internal reflection
            # no refraction
            return False

class Boundary(object):
    """docstring for Boundary"""
    def __init__(self, geometry, n):
        super(Boundary, self).__init__()
        self.geometry = geometry

        self.direction = self.geometry[1, :] - self.geometry[0, :]
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.normal = np.empty(self.direction.shape)
        self.normal[0] = -self.direction[1]
        self.normal[1] = self.direction[0]

        self.n = n # refractive indices

class Environment(object):
    """docstring for Environment"""
    def __init__(self, dxf_file, refractive_indices, scale=1.0):
        super(Environment, self).__init__()

        obstacles = [] # line segments defined by start and end coordinates
        interfaces = [] # boundaries defined by refractive index outside and inside (normal points out)

        drawing = ezdxf.readfile(dxf_file)
        # splines must be exported as line segments
        lines = [(e.dxf.start, e.dxf.end) for e in drawing.query('LINE')]

        while len(lines) > 0:
            obstacle = [lines.pop(0)] # segment in a polygon
            while True:
                try:
                    # find line where start matches previous end
                    line_i = next(i for i, l in enumerate(lines) if l[0] == obstacle[-1][1])
                    line = lines.pop(line_i)
                    obstacle.append(line)
                except StopIteration: # end of polygon
                    break

            # drop z component
            if len(obstacle[0][0]) > 2:
                obstacle = [(start[0:2], end[0:2]) for start, end in obstacle]

            # compute signed area
            # http://mathworld.wolfram.com/PolygonArea.html
            A = 0
            for start, end in obstacle:
                A += start[0]*end[1] - start[1]*end[0]
            if A > 0:
                # reverse order
                obstacle = [(end, start) for start, end in obstacle]

            refractive_index = refractive_indices.pop(0)
            interface = [refractive_index]*len(obstacle) # same for every line segment in obstacle

            # seperate obstacles with a row of nans
            obstacle.append(((np.nan, np.nan), (np.nan, np.nan)))
            interface.append((np.nan, np.nan))
            # and add to mega list
            obstacles += obstacle
            interfaces += interface

            self.obstacles = np.array(obstacles)*scale
            self.interfaces = np.array(interfaces)

    def __getitem__(self, key):
        if type(key) is slice:
            raise TypeError('Slices are not supported')
            return None, None
        if key < 0 or key >= len(self.obstacles):
            raise IndexError('Index out of bounds')
            return None, None

        return Boundary(self.obstacles[key, :, :], self.interfaces[key, :])
