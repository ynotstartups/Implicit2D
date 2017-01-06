def scale_range_to_range(from_min, from_max, to_min, to_max, to_int = False):

    assert from_max > from_min
    assert to_max > to_min

    f = lambda origin_number: ((origin_number-from_min)/(from_max - from_min))*(to_max - to_min) + to_min

    if to_int:
        return lambda origin_number : int(f(origin_number))
    else:
        return f

def using_sdf():
    import json
    import numpy as np
    with open('cat_sdf.txt', 'r') as outfile:
        sdf = json.load(outfile)

    sdf = np.array(sdf)

    print(sdf.shape)

    print(np.max(np.max(sdf)))

    print(np.min(np.min(sdf)))

def test_using_sdf():
    f = scale_range_to_range(0, 254, -1, 1)
    assert isinstance(f(0), float)
    assert f(0) == -1.0
    assert f((254+0)/2) == 0.0
    assert f(254) == 1.0

    f = scale_range_to_range(0, 254, -1, 1, to_int = True)
    assert isinstance(f(0), int)
    assert f(0) == -1
    assert f((254+0)/2) == 0
    assert f(254) == 1

import Implicit2D
import numpy as np
import json

class ImplicitSDF(Implicit2D.ImplicitObject):
    def __init__(self, sdf_filename, xmin, xmax, ymin, ymax):

        assert xmin < xmax, 'incorrect usage xmin {} > xmax {}'.format(xmin, xmax)
        assert ymin < ymax, 'incorrect usage ymin {} > ymax {}'.format(ymin, ymax)

        self.implicit_lambda_function = None
        with open(sdf_filename, 'r') as outfile:
            self.sdf = np.array(json.load(outfile))
            assert len(self.sdf.shape) == 2

        print('which is dim is length which is width')
        self.location_map_x = scale_range_to_range(xmin, xmax, 0, self.get_sdf_height() - 1, to_int=True)
        self.location_map_y = scale_range_to_range(ymin, ymax, 0, self.get_sdf_width() - 1, to_int=True)

        self.sign_distance_map = scale_range_to_range(np.min(np.min(self.sdf)),
                                                      np.max(np.max(self.sdf)),
                                                      -1,
                                                      1)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def get_sdf_width(self):    
        return self.sdf.shape[0]
    def get_sdf_height(self):
        return self.sdf.shape[1]

    def eval_point(self, two_d_point):
        global_x = two_d_point[0][0]
        global_y = two_d_point[1][0]

        assert global_x <= self.xmax, 'global_x {} self.xmax {}'.format(global_x, self.xmax)
        assert global_x >= self.xmin, 'global_x {} self.xmin {}'.format(global_x, self.xmin)
        assert global_y <= self.ymax, 'global_y {} self.ymax {}'.format(global_y, self.ymax)
        assert global_y >= self.ymin, 'global_y {} self.ymin {}'.format(global_y, self.ymin)

        local_x = self.location_map_x(global_x)
        local_y = self.location_map_y(global_y)


        assert isinstance(local_x, int)
        assert isinstance(local_y, int)

        return self.sign_distance_map(self.sdf[local_x][local_y])

def main():

    a = ImplicitSDF('cat_sdf.txt', 0, 1, 0, 1)
    a.visualize_distance_field(0, 1, 0, 1, num_points = 300)

if __name__ == '__main__':
    test_using_sdf()

    main()
















