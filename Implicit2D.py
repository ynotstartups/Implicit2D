'''
Based on https://www.mattkeeter.com/projects/contours/
and http://www.iquilezles.org/
'''
import numpy as np
from util import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
ax1.set_xlim([-0, 1])
ax1.set_ylim([-0, 1])

class ImplicitObject:
    def __init__(self, implicit_lambda_function):
        self.implicit_lambda_function = implicit_lambda_function
        
    def eval_point(self, two_d_point):
        assert two_d_point.shape == (2, 1) # not allow vectorize yet
        value = self.implicit_lambda_function(two_d_point[0][0], two_d_point[1][0])
        return value;
    
    def is_point_inside(self, two_d_point):

        assert two_d_point.shape == (2, 1), "two_d_point format incorrect, {}".format(two_d_point)
        value = self.eval_point(two_d_point)
        if value <= 0:
            return True
        else:
            return False
        
    def union(self, ImplicitObjectInstance):
        return ImplicitObject(lambda x, y: min(
                                              self.eval_point(np.array([[x], [y]])),
                                              ImplicitObjectInstance.eval_point(np.array([[x], [y]]))
                                              ))

    def intersect(self, ImplicitObjectInstance):
        return ImplicitObject(lambda x, y: max(
                                              self.eval_point(np.array([[x], [y]])),
                                              ImplicitObjectInstance.eval_point(np.array([[x], [y]]))
                                              ))
    def negate(self):
        return ImplicitObject(lambda x, y: -1 * self.eval_point(np.array([[x], [y]])))
    
    def substraction(self, ImplicitObjectInstance):
        # substraction ImplicitObjectInstance from self
        return self.intersect(ImplicitObjectInstance.negate())


# distance deformations

    # http://www.iquilezles.org/www/articles/smin/smin.htm
    # exponential smooth min (k = 32);
    # float smin( float a, float b, float k )
    # {
    #     float res = exp( -k*a ) + exp( -k*b );
    #     return -log( res )/k;
    # }

    # http://www.iquilezles.org/www/articles/smin/smin.htm
    # You must be carefull when using distance transformation functions, 
    # as the field created might not be a real distance function anymore. 
    # You will probably need to decrease your step size, 
    # if you are using a raymarcher to sample this.
    # The displacement example below is using sin(20*p.x)*sin(20*p.y)*sin(20*p.z) as displacement pattern,
    # but you can of course use anything you might imagine. 
    # As for smin() function in opBlend(), please read the smooth minimum article in this same site. 

 
    def exponential_smooth_union(self, ImplicitObjectInstance):
        def smin(a, b, smooth_parameter = 32):
            res = np.exp( -smooth_parameter*a ) + np.exp( -smooth_parameter*b );
            return -np.log(res)/smooth_parameter

        return ImplicitObject(lambda x, y: smin(
                                              self.eval_point(np.array([[x], [y]])),
                                              ImplicitObjectInstance.eval_point(np.array([[x], [y]]))
                                              ))

    # // polynomial smooth min (k = 0.1);
    # float smin( float a, float b, float k )
    # {
    #     float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    #     return mix( b, a, h ) - k*h*(1.0-h);
    # }

    def polynomial_smooth_union(self, ImplicitObjectInstance):

        def smin(a, b, smooth_parameter = 0.1):
            h = clamp(0.5+0.5*(b-a)/smooth_parameter, 0.0, 1.0 )
            return mix( b, a, h ) - smooth_parameter*h*(1.0-h);

        return ImplicitObject(lambda x, y: smin(
                                              self.eval_point(np.array([[x], [y]])),
                                              ImplicitObjectInstance.eval_point(np.array([[x], [y]]))
                                              ))

    # cannot make it work
    
    # // power smooth min (k = 8);
    # float smin( float a, float b, float k )
    # {
    #     a = pow( a, k ); b = pow( b, k );
    #     return pow( (a*b)/(a+b), 1.0/k );
    # }

    # def power_smooth_union(self, ImplicitObjectInstance):
    #     def smin(a, b, smooth_parameter=3):
    #         print("---------------------")
    #         print(type(a))
    #         print(type(b))
    #         a = pow( a, smooth_parameter )
    #         b = pow( b, smooth_parameter )

    #         return np.log(a +)
    #         # print(pow( (a*b)/(a+b), 1.0/smooth_parameter ))

    #         # return pow( (a*b)/(a+b), 1.0/smooth_parameter )

    #     return ImplicitObject(lambda x, y: smin(
    #                                           self.eval_point(np.array([[x], [y]])),
    #                                           ImplicitObjectInstance.eval_point(np.array([[x], [y]]))
    #                                           ))

    
    # distance deformations

    # float opDisplace( vec3 p )
    #     {
    #         float d1 = primitive(p);
    #         float d2 = displacement(p);
    #         return d1+d2;
    #     }

    def displace(self, frequency, scale):
        def displacement(x, y, frequency, scale):
            return self.eval_point(np.array([[x], [y]])) + (np.sin(frequency*x)*np.sin(frequency*y))/scale

        return ImplicitObject(lambda x, y: displacement(x, y, frequency, scale))


    # domain deformations


    def derivative_at_point(self, two_d_point, epsilon = 0.001):

        assert two_d_point.shape == (2, 1), 'wrong data two_d_point {}'.format(two_d_point)
        x = two_d_point[0][0]
        y = two_d_point[1][0]

        dx = self.eval_point(np.array([[x + epsilon], [y]])) - self.eval_point(np.array([[x - epsilon], [y]]))
        dy = self.eval_point(np.array([[x], [y + epsilon]])) - self.eval_point(np.array([[x], [y - epsilon]])) 

        length = np.sqrt(dx**2 + dy**2)

        if length <= epsilon:
            print('dodgy: probably error')
            print(two_d_point)
            print(dx)
            print(dy)
            print(self.eval_point(np.array([[x + epsilon], [y]])))
            print(self.eval_point(np.array([[x - epsilon], [y]])))
            print(self.eval_point(np.array([[x], [y + epsilon]])))
            print(self.eval_point(np.array([[x], [y - epsilon]])) )
            return np.array([[0],[0]])
        else:        
            assert length >= epsilon, \
                'length {} if less than epislon {} check dx {} dy {} two_d_point {}'.format(
                    length, epsilon, dx, dy, two_d_point
                )

            return np.array([[dx / length],[dy / length]])

    def visualize_bitmap(self, xmin, xmax, ymin, ymax, num_points=200):
        self.visualize(xmin, xmax, ymin, ymax, 'bitmap', num_points)

    def visualize_distance_field(self, xmin, xmax, ymin, ymax, num_points=200):
        self.visualize(xmin, xmax, ymin, ymax, 'distance_field', num_points)


    def visualize(self, xmin, xmax, ymin, ymax, visualize_type = 'bitmap', num_points=200):
        
        assert xmin!=xmax, "incorrect usage xmin == xmax"
        assert ymin!=ymax, "incorrect usage ymin == ymax"
        assert visualize_type in ['bitmap', 'distance_field'], \
            'visualize_type should be either bitmap or distance_field, but not {}'.format(visualize_type)

        visualize_matrix = np.empty((num_points, num_points));
        
        import matplotlib.pyplot as plt
        x_linspace = np.linspace(xmin, xmax, num_points)
        y_linspace = np.linspace(ymin, ymax, num_points)
        for x_counter in range(len(x_linspace)):
            for y_counter in range(len(y_linspace)):
                x = x_linspace[x_counter]
                y = y_linspace[y_counter]

                if visualize_type == 'bitmap':
                    visualize_matrix[x_counter][y_counter] = \
                        not self.is_point_inside(np.array([[x],[y]])) # for mapping the color of distance_field
                elif visualize_type == 'distance_field':
                    visualize_matrix[x_counter][y_counter] = self.eval_point(np.array([[x],[y]]))
                else:
                    raise ValueError('Unknown visualize_type -> {}'.format(visualize_type))

        visualize_matrix = np.rot90(visualize_matrix)
        assert(visualize_matrix.shape == (num_points, num_points))

        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(visualize_matrix, cmap=plt.cm.gray)
        plt.show() # TODO: label on x, y axis



# In[12]:

class ImplicitCircle(ImplicitObject):
    def __init__(self, x0, y0, r0):
        self.implicit_lambda_function = lambda x, y: np.sqrt((x - x0)**2 + (y - y0)**2) - r0

# In[15]:

class Left(ImplicitObject):
    def __init__(self, x0):
        self.implicit_lambda_function = lambda x, _: x - x0
        
class Right(ImplicitObject):
    def __init__(self, x0):
        self.implicit_lambda_function = lambda x, _: x0 - x
        
class Lower(ImplicitObject):
    def __init__(self, y0):
        self.implicit_lambda_function = lambda _, y: y - y0
        
class Upper(ImplicitObject):
    def __init__(self, y0):
        self.implicit_lambda_function = lambda _, y: y0 - y

# In[17]:

class ImplicitRectangle(ImplicitObject):
    def __init__(self, xmin, xmax, ymin, ymax):
        assert xmin!=xmax, "incorrect usage xmin == xmax"
        assert ymin!=ymax, "incorrect usage ymin == ymax"
        # right xmin ∩ left xmax ∩ upper ymin ∩ lower ymax
        self.implicit_lambda_function = (Right(xmin).intersect(Left(xmax)).intersect(Upper(ymin)).intersect(Lower(ymax))).implicit_lambda_function


class ImplicitFailureStar(ImplicitObject):
    # http://www.iquilezles.org/live/index.htm
    def __init__(self, inner_radius, outer_radius, frequency, x0=0, y0=0):
        self. implicit_lambda_function = \
            lambda x, y:inner_radius + outer_radius*np.cos(np.arctan2(y,x)*frequency)

class ImplicitStar(ImplicitObject):
    # http://www.iquilezles.org/live/index.htm
    def __init__(self, inner_radius, outer_radius, frequency, x0=0, y0=0):
        self. implicit_lambda_function = \
            lambda x, y: ImplicitStar.smoothstep(
                            inner_radius + outer_radius*np.cos(np.arctan2(y - x0, x - y0)*frequency), 
                            inner_radius + outer_radius*np.cos(np.arctan2(y - x0, x - y0)*frequency) + 0.01,
                            np.sqrt((x - x0)**2 + (y - y0)**2)
                        )

    @staticmethod
    def smoothstep(edge0, edge1, x):
        # https://en.wikipedia.org/wiki/Smoothstep
        #     float smoothstep(float edge0, float edge1, float x)
        # {
        #     // Scale, bias and saturate x to 0..1 range
        #     x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0); 
        #     // Evaluate polynomial
        #     return x*x*(3 - 2*x);
        # }

        x = clamp((x - edge0)/(edge1 -edge0), 0.0, 1.0)
        return x*x*(3-2*x)

class ImplicitTree(ImplicitStar):
    # http://www.iquilezles.org/live/index.htm
    def __init__(self, inner_radius=0.2, outer_radius=0.1, frequency=10, x0=0.4, y0=0.5):

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.frequency = frequency
        self.x0 = x0
        self.y0 = y0

        self. implicit_lambda_function = self.implicit_lambda_function


    def implicit_lambda_function(self, x, y):

        local_x = x - self.x0 
        local_y = y - self.y0

        r = self.inner_radius + self.outer_radius*np.cos(np.arctan2(local_y, local_x)*self.frequency + 20*local_x + 1)
        result = ImplicitStar.smoothstep(r, r + 0.01,np.sqrt(local_x**2 + local_y**2))
        r = 0.015
        r += 0.002 * np.cos(120.0 * local_y)
        r += np.exp(-20.0 * y)
        result *= 1.0 - (1.0 - ImplicitStar.smoothstep(r, r + 1, abs(local_x+ 0.2*np.sin(2.0 *local_y)))) * \
                              (1.0 - ImplicitStar.smoothstep(0.0, 0.1, local_y))
        return result


# In[19]:

#     h = (rectangle (0.1, 0.1) (0.25, 0.9) ∪
#          rectangle (0.1, 0.1) (0.6, 0.35) ∪
#          circle (0.35, 0.35) 0.25) ∩ inv
#          (circle (0.35, 0.35) 0.1 ∪
#           rectangle (0.25, 0.1) (0.45, 0.35))
#         i = rectangle (0.75, 0.1) (0.9, 0.55) ∪
#         circle (0.825, 0.75) 0.1
        

class Tree:
    def __init__(self,
                 tree_or_cell_0, tree_or_cell_1, tree_or_cell_2, tree_or_cell_3):
        
        assert (isinstance(tree_or_cell_0, Tree) | isinstance(tree_or_cell_0, Cell)) &                (isinstance(tree_or_cell_1, Tree) | isinstance(tree_or_cell_1, Cell)) &                (isinstance(tree_or_cell_2, Tree) | isinstance(tree_or_cell_2, Cell)) &                (isinstance(tree_or_cell_3, Tree) | isinstance(tree_or_cell_3, Cell))
            
        self.tree_or_cell_0 = tree_or_cell_0
        self.tree_or_cell_1 = tree_or_cell_1
        self.tree_or_cell_2 = tree_or_cell_2
        self.tree_or_cell_3 = tree_or_cell_3


class Cell:
    def __init__(self, xmin, xmax, ymin, ymax, cell_type):
        assert xmin!=xmax, "incorrect usage xmin == xmax"
        assert ymin!=ymax, "incorrect usage ymin == ymax"
        assert cell_type in ['Empty', 'Full', 'Leaf', 'Root', 'NotInitialized']
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xmid = (xmin + xmax)/2
        self.ymid = (ymin + ymax)/2
        
        self.cell_type = cell_type
        '''
        2 -- 3
        |    |
        0 -- 1
        '''
        self.point_0 = np.array([[xmin],[ymin]])
        self.point_1 = np.array([[xmax],[ymin]])
        self.point_2 = np.array([[xmin],[ymax]])
        self.point_3 = np.array([[xmax],[ymax]])
        
        if self.is_Root():
            self.to_Root()
        
    def xmin_xmax_ymin_ymax(self):
        return [self.xmin, self.xmax, self.ymin, self.ymax]
    
    def to_Root(self):
        assert self.cell_type != 'Root'
        self.cell_type = 'Root'
        self.cell0 = Cell(self.xmin, self.xmid, self.ymin, self.ymid, 'NotInitialized')
        self.cell1 = Cell(self.xmid, self.xmax, self.ymin, self.ymid, 'NotInitialized')
        self.cell2 = Cell(self.xmin, self.xmid, self.ymid, self.ymax, 'NotInitialized')
        self.cell3 = Cell(self.xmid, self.xmax, self.ymid, self.ymax, 'NotInitialized')
        
    def to_Empty(self):
        assert self.is_Root()
        self.cell_type = 'Empty'
        
        del self.cell0
        del self.cell1
        del self.cell2
        del self.cell3

    def to_Full(self):
        assert self.is_Root()
        self.cell_type = 'Full'

        del self.cell0
        del self.cell1
        del self.cell2
        del self.cell3

    def to_Leaf(self):
        assert self.is_Root()
        self.cell_type = 'Leaf'

        del self.cell0
        del self.cell1
        del self.cell2
        del self.cell3

    def check_not_initialized_exists(self):
        # raise Error if NotInitialized exists
        if self.is_Root():
            self.cell0.check_not_initialized_exists()
            self.cell1.check_not_initialized_exists()
            self.cell2.check_not_initialized_exists()
            self.cell3.check_not_initialized_exists()
        else:
            if self.is_NotInitialized():
                raise ValueError('cell should not be as cell_type {}'.format(self.cell_type))
            else:
                pass

    def check_Leaf_exists(self):
        # raise Error if NotInitialized exists

        if self.is_Root():
            self.cell0.check_Leaf_exists()
            self.cell1.check_Leaf_exists()
            self.cell2.check_Leaf_exists()
            self.cell3.check_Leaf_exists()
        else:
            if self.is_Leaf():
                raise ValueError('cell should not be as cell_type {}'.format(self.cell_type))
            else:
                pass

    def eval_type(self, implicit_object_instance):
        assert self.is_NotInitialized(), 'this function is only called when the cell type is not initialized'
        is_point0_inside = implicit_object_instance.is_point_inside(self.point_0)
        is_point1_inside = implicit_object_instance.is_point_inside(self.point_1)
        is_point2_inside = implicit_object_instance.is_point_inside(self.point_2)
        is_point3_inside = implicit_object_instance.is_point_inside(self.point_3)
        
        if ((is_point0_inside is True) &
           (is_point1_inside is True) &
           (is_point2_inside is True) &
           (is_point3_inside is True) ):
                
            self.cell_type = 'Full'
        
        elif ( (is_point0_inside is False) &
           (is_point1_inside is False) &
           (is_point2_inside is False) &
           (is_point3_inside is False) ):
                
            self.cell_type = 'Empty'
        
        else:
            # print('to Leaf')
            self.cell_type = 'Leaf'

    def add_marching_cude_points(self, edge_vectice_0, edge_vectice_1, mc_connect_indicator):
        assert self.is_Leaf()
        # self.edge_vectice_0 = edge_vectice_0
        # self.edge_vectice_1 = edge_vectice_1
        try:
            self.edge_vectices.append((edge_vectice_0, edge_vectice_1, mc_connect_indicator))
        except AttributeError:
            self.edge_vectices = [(edge_vectice_0, edge_vectice_1, mc_connect_indicator)]

    def get_first_indicator(self):
        assert self.is_Leaf()
        assert len(self.edge_vectices) >= 1
        return self.edge_vectices[0][2]

    def get_second_indicator(self):
        assert self.is_Leaf()
        assert len(self.edge_vectices) >= 1
        print(self.edge_vectices[1])
        print(self.edge_vectices[1][0])

        print(self.edge_vectices[1][1])

        return self.edge_vectices[1][2]

    def get_marching_cude_points(self):
        # remove the edges = 
        raise NotImplementedError

    def debug_print(self, counter):
        counter += 1
        if self.cell_type in ['Full', 'Empty', 'Leaf', 'NotInitialized']:
            # print(counter)
            pass
        else:
            self.cell0.debug_print(counter)
            self.cell1.debug_print(counter)
            self.cell2.debug_print(counter)
            self.cell3.debug_print(counter)
    
    def visualize(self, ax1):



        if self.cell_type in ['Empty', 'Full', 'Leaf', 'NotInitialized']:
            if self.is_Empty():
                color = 'grey'
            elif self.is_Full():
                color = 'black'
            elif self.is_Leaf():
                color = 'green'
            elif self.is_NotInitialized():
                color = 'red'
            else:
                raise ValueError('cell should not be as cell_type {}'.format(self.cell_type))
                
            ax1.add_patch(
                patches.Rectangle(
                    (self.xmin, self.ymin),   # (x,y)
                    self.xmax - self.xmin,    # width
                    self.ymax - self.ymin,    # height
                    edgecolor = color,
                    facecolor = 'white'
                )
            )
        elif self.is_Root():
            self.cell0.visualize(ax1)
            self.cell1.visualize(ax1)
            self.cell2.visualize(ax1)
            self.cell3.visualize(ax1)

        else:
            raise ValueError('cell should not be as cell_type {}'.format(self.cell_type))
    
    def print_type(self):
        if not self.is_Root():
            print(self.cell_type)
        else:
            print('Root')
            self.cell0.print_type()
            self.cell1.print_type()
            self.cell2.print_type()
            self.cell3.print_type()



    def initialise_cell_type(self, implicit_object_instance):

        if self.is_Root():
            self.cell0.initialise_cell_type(implicit_object_instance)
            self.cell1.initialise_cell_type(implicit_object_instance)
            self.cell2.initialise_cell_type(implicit_object_instance)
            self.cell3.initialise_cell_type(implicit_object_instance)
        elif self.is_NotInitialized():
            self.eval_type(implicit_object_instance)
        else:
            raise ValueError('There should not be any other \
                              cell_type when calling this function -> {}'.format(self.cell_type))


    def bisection(self, two_points_contain_vectice, implicit_object_instance, epsilon = 0.0001):

        ''' not considering the orientation left_or_right '''

        assert len(two_points_contain_vectice[0]) == 2, "two_points_contain_vectice[0] wrong format, {}".format(two_points_contain_vectice[0])
        assert len(two_points_contain_vectice[1]) == 2, "two_points_contain_vectice[0] wrong format, {}".format(two_points_contain_vectice[0])
        assert isinstance(two_points_contain_vectice, np.ndarray), 'two_points_contain_vectice has wrong type {}'.format(type(two_points_contain_vectice))

        #two_points_contain_vectice =  [[[ 0.125  ]
        #   [ 0.09375]]

        #  [[ 0.125  ]
        #   [ 0.125  ]]]


        edge0_x = two_points_contain_vectice[0][0][0]
        edge0_y = two_points_contain_vectice[0][1][0]
        edge1_x = two_points_contain_vectice[1][0][0]
        edge1_y = two_points_contain_vectice[1][1][0]

        # print(edge0_x)
        # print(edge0_y)
        # print(edge1_x)
        # print(edge1_y)


        is_edge0_inside = implicit_object_instance.is_point_inside(np.array([[edge0_x], [edge0_y]]))
        is_edge1_inside = implicit_object_instance.is_point_inside(np.array([[edge1_x], [edge1_y]]))

        
        # TODO: find a assert to make sure two_points_contain_vectice are not the same

        assert is_edge0_inside != is_edge1_inside,\
                'it cannot be both points {} {}'.format(
                    is_edge0_inside,
                    is_edge1_inside
                )


        edge_xmid = (edge1_x + edge0_x)/2
        edge_ymid = (edge1_y + edge0_y)/2

        if np.sqrt((edge1_x - edge0_x)**2 + (edge1_y - edge0_y)**2) <= epsilon:
            return (edge_xmid, edge_ymid)

        is_edge_mid_inside = implicit_object_instance.is_point_inside(np.array([[edge_xmid], [edge_ymid]]))

        if is_edge_mid_inside is not is_edge0_inside:
            return self.bisection(np.array([[[edge0_x], [edge0_y]],[[edge_xmid], [edge_ymid]]]),
                                        implicit_object_instance,
                                        epsilon = 0.01)

        elif is_edge_mid_inside is not is_edge1_inside:
            return self.bisection(np.array([[[edge1_x], [edge1_y]],[[edge_xmid], [edge_ymid]]]),
                                        implicit_object_instance,
                                        epsilon = 0.01)        
        else:
            raise ValueError

    @staticmethod
    def mc_two_one_connect_h(two_vertice_first, two_type_first, two_vertice_second, two_type_second, 
                             one_vertice, one_type):
        # connect left to right two to one
        assert 'left' in one_type, 'left not in one_type {}'.format(one_type)

        if 'left' in one_type[0]:
            if two_type_first == ['bottom', 'right']:
                return np.array([two_vertice_first, one_vertice])
            elif two_type_second == ['bottom', 'right']:
                return np.array([two_vertice_second, one_vertice])
            else:
                raise ValueError('two_type_first {}, two_type_second {}'.format(two_type_first, two_type_second))
        elif 'left' in one_type[1]:
            if two_type_first == ['right', 'top']:
                return np.array([two_vertice_first, one_vertice])
            elif two_type_second == ['right', 'top']:
                return np.array([two_vertice_second, one_vertice])
            else:
                raise ValueError('two_type_first {}, two_type_second {}'.format(two_type_first, two_type_second))
        else:
            raise ValueError

    @staticmethod
    def mc_one_two_connect_h(one_vertice, one_type, 
                             two_vertice_first, two_type_first, two_vertice_second, two_type_second):
        # connect left to right one to two
        assert 'right' in one_type, 'right not in one_type {}'.format(one_type)
        if 'right' in one_type[0]:
            if two_type_first == ['top', 'left']:
                return np.array([one_vertice, two_vertice_first])
            elif two_type_second == ['top', 'left']:
                return np.array([one_vertice, two_vertice_second])
            else:
                raise ValueError('two_type_first {}, two_type_second {}'.format(two_type_first, two_type_second))
        elif 'right' in one_type[1]:
            if two_type_first == ['left', 'bottom']:
                return np.array([one_vertice, two_vertice_first])
            elif two_type_second == ['left', 'bottom']:
                return np.array([one_vertice, two_vertice_second])
            else:
                raise ValueError('two_type_first {}, two_type_second {}'.format(two_type_first, two_type_second))
        else:
            raise ValueError

    @staticmethod
    def mc_two_two_connect_h(two_vertice_l_first, two_type_l_first, two_vertice_l_second, two_type_l_second,
                             two_vertice_r_first, two_type_r_first, two_vertice_r_second, two_type_r_second):
        
        # connect left to right two to two
        ''' not tested '''

        if two_type_l_first == ('top', 'left'):
            assert two_type_l_second == ('bottom', 'right')

            if two_type_r_first == ('bottom', 'left'):
                return np.array([two_type_l_second, two_vertice_r_first])
            elif two_type_r_second == ('bottom', 'left'):
                return np.array([two_type_l_second, two_vertice_r_second])
            else:
                raise ValueError

        elif two_type_l_first == ('bottom', 'right'):
            assert two_type_l_second == ('top', 'left')
            if two_type_r_first == ('bottom', 'left'):
                return np.array([two_vertice_l_first, two_vertice_r_first])
            elif two_type_r_second == ('bottom', 'left'):
                return np.array([two_vertice_l_first, two_vertice_r_second])
            else:
                raise ValueError

        elif two_type_l_first == ('left', 'bottom'):
            assert two_type_l_second == ('right', 'top')

            if two_type_r_first == ('top', 'left'):
                return np.array([two_type_l_second, two_type_r_first])
            elif two_type_r_second == ('top', 'left'):
                return np.array([two_type_l_second, two_type_r_second])
            else:
                return ValueError

        elif two_type_l_first == ('right', 'top'):
            assert two_type_l_second == ('left', 'bottom')

            if two_type_r_first == ('top', 'left'):
                return np.array([two_type_l_first, two_type_r_first])
            elif two_type_r_second == ('top', 'left'):
                return np.array([two_type_l_first, two_type_r_second])
            else:
                return ValueError
        else:
            raise ValueError

    @staticmethod
    def mc_two_one_connect_v(two_vertice_first, two_type_first, two_vertice_second, two_type_second, 
                             one_vertice, one_type):
        # connect top to bottom two to one
        assert 'top' in one_type, 'bottom not in one_type {}'.format(one_type)

        if 'top' in one_type[0]: # TODO: why not == instead of in
            if two_type_first == ['left', 'bottom']:
                return np.array([two_vertice_first, one_vertice])
            elif two_type_second == ['left', 'bottom']:
                return np.array([two_type_second, one_vertice])
            else:
                raise ValueError

        elif 'top' in one_type[1]:
            if two_type_first == ['bottom', 'right']:
                return np.array([two_vertice_first, one_vertice])
            elif two_type_second == ['bottom', 'right']:
                return np.array([two_type_second, one_vertice])
            else:
                raise ValueError

        else:
            raise ValueError

    @staticmethod
    def mc_one_two_connect_v(one_vertice, one_type, 
                             two_vertice_first, two_type_first, two_vertice_second, two_type_second):
        # connect top to bottom two to one
        assert 'bottom' in one_type

        if 'bottom' in one_type[0]: # TODO: why not == instead of in
            if two_type_first == ['right', 'top']:
                return np.array([one_vertice, two_vertice_first])
            elif two_type_second == ['right', 'top']:
                return np.array([one_vertice, two_vertice_second])
            else:
                raise ValueError

        elif 'bottom' in one_type[1]:
            if two_type_first == ['top', 'left']:
                return np.array([one_vertice, two_vertice_first])
            elif two_type_second == ['top', 'left']:
                return np.array([one_vertice, two_vertice_second])
            else:
                raise ValueError

        else:
            raise ValueError

    @staticmethod
    def mc_two_two_connect_v(two_vertice_t_first, two_type_t_first, two_vertice_t_second, two_type_t_second,
                             two_vertice_b_first, two_type_b_first, two_vertice_b_second, two_type_b_second):
        
        ''' not tested '''
        if two_type_t_first == ['top', 'left']:
            assert two_type_t_second == ['bottom', 'right']

            if two_type_b_first == ['right', 'top']:
                return np.array([two_type_t_second, two_type_b_first])
            elif two_type_b_second == ['right', 'top']:
                return np.array([two_type_t_second, two_type_b_second])
            else:
                raise ValueError

        elif two_type_t_first == ['bottom', 'right']:
            assert two_type_t_second == ['top', 'left']

            if two_type_b_first == ['right', 'top']:
                return np.array([two_type_t_first, two_type_b_first])
            elif two_type_b_second == ['right', 'top']:
                return np.array([two_type_t_first, two_type_b_second])
            else:
                raise ValueError

        elif two_type_t_first == ['right', 'top']:
            assert two_type_t_second == ['left', 'bottom']

            if two_type_b_first == ['top', 'left']:
                return np.array([two_type_t_second, two_type_b_first])
            elif two_type_b_second == ['top', 'left']:
                return np.array([two_type_t_second, two_type_b_second])
            else:
                raise ValueError
        elif two_type_t_first == ['left', 'bottom']:
            assert two_type_t_second == ['right', 'top']

            if two_type_b_first == ['top', 'left']:
                return np.array([two_type_t_first, two_type_b_first])
            elif two_type_b_second == ['top', 'left']:
                return np.array([two_type_t_first, two_type_b_second])
            else:
                raise ValueError

        else:
            raise ValueError


    def marching_cube(self, implicit_object_instance, edges):
        
        def find_add_mc_edges_vertices(self, points_for_edges, edges):
            two_points_contain_vectice_0 = np.array(points_for_edges[0])
            two_points_contain_vectice_1 = np.array(points_for_edges[1])
            mc_connect_indicator = np.array(points_for_edges[2])

            edge_vectice_0 = self.bisection(two_points_contain_vectice_0, implicit_object_instance)
            edge_vectice_1 = self.bisection(two_points_contain_vectice_1, implicit_object_instance)

            self.add_marching_cude_points(edge_vectice_0, edge_vectice_1, mc_connect_indicator)

            edges.append([edge_vectice_0, edge_vectice_1])

            return [edge_vectice_0, edge_vectice_1]

        self.check_not_initialized_exists() # self testing
        
        # 0 to the left, 1 to the right
        marching_cube_edge_indicator = {
            (False, False, False, False): np.array([]),
            (False, False, True, False): np.array([[self.point_0, self.point_2],[self.point_2, self.point_3], [('top','left')]]),
            (False, False, False, True): np.array([[self.point_1, self.point_3],[self.point_2, self.point_3], [('right','top')]]),
            (False, False, True, True): np.array([[self.point_0, self.point_2],[self.point_1, self.point_3], [('right','left')]]),
            (False, True, False, False): np.array([[self.point_0, self.point_1],[self.point_1, self.point_3], [('bottom','right')]]),
            (False, True, True, False): np.array([[self.point_0, self.point_1],[self.point_1, self.point_3], [('bottom','right')],
                                                  [self.point_0, self.point_2],[self.point_2, self.point_3], [('top','left')]]),
            (False, True, False, True): np.array([[self.point_0, self.point_1], [self.point_2, self.point_3],  [('bottom','top')]]),
            (False, True, True, True): np.array([[self.point_0, self.point_1], [self.point_0, self.point_2], [('bottom','left')]]),
            (True, False, False, False): np.array([[self.point_0, self.point_1], [self.point_0, self.point_2], [('left','bottom')]]),
            (True, False, True, False):  np.array([[self.point_0, self.point_1], [self.point_2, self.point_3], [('top','bottom')]]),
            (True, False, False, True): np.array([[self.point_0, self.point_1],[self.point_0, self.point_2], [('left','bottom')],
                                                  [self.point_1, self.point_3],[self.point_2, self.point_3], [('right','top')]]),
            (True, False, True, True): np.array([[self.point_0, self.point_1],[self.point_1, self.point_3], [('right','bottom')]]),
            (True, True, False, False): np.array([[self.point_0, self.point_2],[self.point_1, self.point_3], [('left','right')]]),
            (True, True, True, False): np.array([[self.point_1, self.point_3],[self.point_2, self.point_3], [('top','right')]]),
            (True, True, False, True): np.array([[self.point_0, self.point_2],[self.point_2, self.point_3], [('left', 'top')]]),
            (True, True, True, True):np.array([])


        }
                

        if self.is_Leaf():
            
            # repeated code
            is_point0_inside = implicit_object_instance.is_point_inside(self.point_0)
            is_point1_inside = implicit_object_instance.is_point_inside(self.point_1)
            is_point2_inside = implicit_object_instance.is_point_inside(self.point_2)
            is_point3_inside = implicit_object_instance.is_point_inside(self.point_3)

            points_for_edges = marching_cube_edge_indicator[(is_point0_inside, is_point1_inside, is_point2_inside, is_point3_inside)]
            
            if len(points_for_edges) == 0:
                pass
            elif len(points_for_edges) == 3: # one edge

                # two_points_contain_vectice_0 = np.array(points_for_edges[0])
                # two_points_contain_vectice_1 = np.array(points_for_edges[1])

                # edge_vectice_0 = self.bisection(two_points_contain_vectice_0, implicit_object_instance)
                # edge_vectice_1 = self.bisection(two_points_contain_vectice_1, implicit_object_instance)

                # self.add_marching_cude_points(edge_vectice_0, edge_vectice_1)

                # edges.append([edge_vectice_0, edge_vectice_1])
                find_add_mc_edges_vertices(self, points_for_edges, edges)

            elif len(points_for_edges) == 6: # two edges

                # two_points_contain_vectice_0 = np.array(points_for_edges[0])
                # two_points_contain_vectice_1 = np.array(points_for_edges[1])

                # edge_vectice_0 = self.bisection(two_points_contain_vectice_0, implicit_object_instance)
                # edge_vectice_1 = self.bisection(two_points_contain_vectice_1, implicit_object_instance)

                # self.add_marching_cude_points(edge_vectice_0, edge_vectice_1)
                # edges.append([edge_vectice_0, edge_vectice_1])

                print('two edges--')

                points_for_edge = points_for_edges[:3]
                assert len(points_for_edge) == 3
                res  = find_add_mc_edges_vertices(self, points_for_edge, edges)
                print(res)
                # two_points_contain_vectice_3 = np.array(points_for_edges[3])
                # two_points_contain_vectice_4 = np.array(points_for_edges[4])

                # edge_vectice_3 = self.bisection(two_points_contain_vectice_3, implicit_object_instance)
                # edge_vectice_4 = self.bisection(two_points_contain_vectice_4, implicit_object_instance)

                # self.add_marching_cude_points(edge_vectice_3, edge_vectice_4)

                # edges.append([edge_vectice_3, edge_vectice_4])

                points_for_edge = points_for_edges[3:]
                assert len(points_for_edge) == 3
                res = find_add_mc_edges_vertices(self, points_for_edge, edges)
                print(res)

            else:
                raise ValueError('there should not be another value...')
        
        elif self.is_Root():
            self.cell0.marching_cube(implicit_object_instance, edges)
            self.cell1.marching_cube(implicit_object_instance, edges)
            self.cell2.marching_cube(implicit_object_instance, edges)
            self.cell3.marching_cube(implicit_object_instance, edges)
        else:
            pass

        return edges

    def interpolate(self, two_d_point,
                    implicit_object_instance):

        assert self.is_Root()

        x_interpolate = two_d_point[0][0]
        y_interpolate = two_d_point[1][0]

        assert self.xmin <= x_interpolate <= self.xmax
        assert self.ymin <= y_interpolate <= self.ymax

        dx = (x_interpolate - self.xmin) / (self.xmax - self.xmin)
        dy = (y_interpolate - self.ymin) / (self.ymax - self.ymin)
        ab = implicit_object_instance.eval_point(np.array([[self.xmin], [self.ymin]])) * (1 - dx) + \
             implicit_object_instance.eval_point(np.array([[self.xmax], [self.ymin]])) * dx
        cd = implicit_object_instance.eval_point(np.array([[self.xmin], [self.ymax]])) * (1 - dx) + \
             implicit_object_instance.eval_point(np.array([[self.xmax], [self.ymax]])) * dx

        return ab * (1 - dy) + cd * dy

    def sampled_distance_fields(self, implicit_object_instance, epsilon = 0.001):

        assert isinstance(implicit_object_instance, ImplicitObject)


        def does_pass_score_test():     
            # using notation https://www.mattkeeter.com/projects/contours/
            exact_i = implicit_object_instance.eval_point(np.array([[self.cell0.xmax], [self.cell0.ymax]]))
            exact_s = implicit_object_instance.eval_point(np.array([[self.cell0.xmin], [self.cell0.ymax]]))
            exact_q = implicit_object_instance.eval_point(np.array([[self.cell0.xmax], [self.cell0.ymin]]))
            exact_r = implicit_object_instance.eval_point(np.array([[self.cell1.xmax], [self.cell1.ymax]]))
            exact_t = implicit_object_instance.eval_point(np.array([[self.cell2.xmax], [self.cell2.ymax]]))

            approximate_i = self.interpolate(np.array([[self.cell0.xmax], [self.cell0.ymax]]), implicit_object_instance)
            approximate_s = self.interpolate(np.array([[self.cell0.xmin], [self.cell0.ymax]]), implicit_object_instance)
            approximate_q = self.interpolate(np.array([[self.cell0.xmax], [self.cell0.ymin]]), implicit_object_instance)
            approximate_r = self.interpolate(np.array([[self.cell1.xmax], [self.cell1.ymax]]), implicit_object_instance)
            approximate_t = self.interpolate(np.array([[self.cell2.xmax], [self.cell2.ymax]]), implicit_object_instance)

            score_all = np.abs(np.array([approximate_i - exact_i,
                                  approximate_s - exact_s, 
                                  approximate_q - exact_q, 
                                  approximate_r - exact_r, 
                                  approximate_t - exact_t, 
                                 ]))
            if np.all(score_all <= epsilon):
                return True
            else:
                return False

        if self.is_Root():

            self.cell0.sampled_distance_fields(implicit_object_instance, epsilon)
            self.cell1.sampled_distance_fields(implicit_object_instance, epsilon)
            self.cell2.sampled_distance_fields(implicit_object_instance, epsilon)
            self.cell3.sampled_distance_fields(implicit_object_instance, epsilon)

            cells_type = [self.cell0.cell_type,
                          self.cell1.cell_type,
                          self.cell2.cell_type,
                          self.cell3.cell_type]

            if  (cells_type.count('Leaf') >= 1 ) and \
                (cells_type.count('Leaf') + cells_type.count('Empty') == 4):
                if does_pass_score_test():
                    self.to_Leaf()

            elif cells_type.count('Empty') == 4:
                if does_pass_score_test():
                    self.to_Empty()

            elif cells_type.count('Full') == 4:
                if does_pass_score_test():
                    self.to_Full()

            else:
                pass

        else:
            pass

    def is_Root(self):
        return self.cell_type == 'Root'
    def is_Leaf(self):
        return self.cell_type == 'Leaf'
    def is_Empty(self):
        return self.cell_type == 'Empty'
    def is_NotInitialized(self):
        return self.cell_type == 'NotInitialized'
    def is_Full(self):
        return self.cell_type == 'Full'

    def add_dual_edge(self, dual_edge):
        assert self.is_Root()
        try:
            self.dual_edges.append(dual_edge)
            print(self.dual_edges)
            if len(self.dual_edges) > 2:
                print('len(self.dual_edges) is {}, large then 2'.format(len(self.dual_edges)))

            # assert len(self.dual_edges) <= 2, 'len(self.dual_edges) is {}, large then 2'.format(len(self.dual_edges))
        except AttributeError:
            self.dual_edges = [dual_edge]

    def has_dual_edge(self):
        assert self.is_Root()
        return hasattr(self, 'dual_edges')

    def get_dual_edge(self):
        return self.dual_edges

    def edgeProcH(self, horizontal_cell_left, horizontal_cell_right):
        if horizontal_cell_left.is_Root() and horizontal_cell_right.is_Root():
            self.edgeProcH(horizontal_cell_left.cell3, horizontal_cell_right.cell2)
            self.edgeProcH(horizontal_cell_left.cell1, horizontal_cell_right.cell0)

        elif horizontal_cell_left.is_Root() and horizontal_cell_right.is_Leaf():
            self.edgeProcH(horizontal_cell_left.cell3, horizontal_cell_right)
            self.edgeProcH(horizontal_cell_left.cell1, horizontal_cell_right)

        elif horizontal_cell_left.is_Leaf() and horizontal_cell_right.is_Root():
            self.edgeProcH(horizontal_cell_left, horizontal_cell_right.cell2)
            self.edgeProcH(horizontal_cell_left, horizontal_cell_right.cell0)

        elif  horizontal_cell_left.is_Leaf() and horizontal_cell_right.is_Leaf():

            if horizontal_cell_left.num_dual_vertex() == 1 and \
               horizontal_cell_right.num_dual_vertex() == 1:

                if 'right' in horizontal_cell_left.get_first_indicator() and \
                  'left' in horizontal_cell_right.get_first_indicator():
                   self.add_dual_edge([horizontal_cell_left.get_first_dual_vertex().flatten().tolist(),
                                        horizontal_cell_right.get_first_dual_vertex().flatten().tolist()]);
                else:
                    pass

            elif horizontal_cell_left.num_dual_vertex() == 2 and \
                 horizontal_cell_right.num_dual_vertex() == 1:
                print('here h 2 1 ')


                left_first_d_v = horizontal_cell_left.get_first_dual_vertex().flatten().tolist()
                left_second_d_v = horizontal_cell_left.get_second_dual_vertex().flatten().tolist()
                left_first_indicator = horizontal_cell_left.get_first_indicator().flatten().tolist()
                left_second_indicator = horizontal_cell_left.get_second_indicator().flatten().tolist()
                right_d_v = horizontal_cell_right.get_first_dual_vertex().flatten().tolist()
                right_indicator = horizontal_cell_right.get_first_indicator().flatten().tolist()

                result = self.mc_two_one_connect_h(left_first_d_v, left_first_indicator, left_second_d_v, left_second_indicator,
                                            right_d_v, right_indicator)

                if result is not None:
                    assert result.shape == (2, 2)
                    print('------here h 2 1  edge----')
                    print(result.tolist())
                    self.add_dual_edge(result.tolist());

                '''
                left_first_d_v = horizontal_cell_left.get_first_dual_vertex()
                left_second_d_v = horizontal_cell_left.get_second_dual_vertex()
                left_first_indicator = horizontal_cell_left.get_first_indicator()
                left_second_indicator = horizontal_cell_left.get_second_indicator()

                right_d_v = horizontal_cell_right.get_first_dual_vertex()
                right_indicator = horizontal_cell_right.get_first_indicator()

                assert 'left' in right_indicator 
                assert not (('right' in left_first_indicator) and 'right' in left_second_indicator)

                if not ('left' in right_indicator):
                    # right cell is not connect to left
                    return None
                elif 'right' in left_first_indicator:
                    self.add_dual_edge([left_first_d_v.flatten().tolist(),
                                        right_d_v.flatten().tolist()]);
                elif 'right' in left_second_indicator:
                    self.add_dual_edge([left_second_d_v.flatten().tolist(),
                                        right_d_v.flatten().tolist()]);
                else:
                    # no right in indicator, dont connect
                    return None
                    # raise ValueError('no right in indicator \
                    #     left_first_indicator {}, left_second_indicator {}'.format(
                    #     left_first_indicator, left_second_indicator
                    #     )
                    # )

                '''

                # if np.linalg.norm(left_first_d_v - right_d_v) <= np.linalg.norm(left_second_d_v - right_d_v):
                #     self.add_dual_edge([left_first_d_v.flatten().tolist(),
                #                         right_d_v.flatten().tolist()]);
                # else:
                #     self.add_dual_edge([left_second_d_v.flatten().tolist(),
                #                         right_d_v.flatten().tolist()]);

            elif horizontal_cell_left.num_dual_vertex() == 1 and \
                 horizontal_cell_right.num_dual_vertex() == 2:

                left_d_v = horizontal_cell_left.get_first_dual_vertex().flatten().tolist()
                left_indicator = horizontal_cell_left.get_first_indicator().flatten().tolist()

                right_first_d_v = horizontal_cell_right.get_first_dual_vertex().flatten().tolist()
                right_first_indicator = horizontal_cell_right.get_first_indicator().flatten().tolist()

                right_second_d_v = horizontal_cell_right.get_second_dual_vertex().flatten().tolist()
                right_second_indicator = horizontal_cell_right.get_second_indicator().flatten().tolist()



                result = self.mc_one_two_connect_h(left_d_v, left_indicator,
                                                    right_first_d_v, right_first_indicator, right_second_d_v, right_second_indicator)
                if result is not None:
                    assert result.shape == (2, 2), 'result {}'.format(result)
                    print('------here h 1 2  edge----')

                    print(left_d_v)
                    print(left_indicator)
                    print(right_first_d_v)
                    print(right_first_indicator)
                    print(right_second_d_v)
                    print(right_second_indicator)
                    print(result.tolist())
                    self.add_dual_edge(result.tolist());

                '''
                print('here h 1 2')

                left_d_v = horizontal_cell_left.get_first_dual_vertex()
                left_indicator = horizontal_cell_left.get_first_indicator()

                right_first_d_v = horizontal_cell_right.get_first_dual_vertex()
                right_second_d_v = horizontal_cell_right.get_second_dual_vertex()

                right_first_indicator = horizontal_cell_right.get_first_indicator()
                right_second_indicator = horizontal_cell_right.get_second_indicator()


                print(left_d_v)
                print(left_indicator)
                print(right_first_d_v)
                print(right_second_d_v)
                print(right_first_indicator)
                print(right_second_indicator)


                if not ('right' in left_indicator):
                    # left is not connect to right
                    return None
                elif 'left' in right_first_indicator:
                    self.add_dual_edge([left_d_v.flatten().tolist(),
                                        right_first_d_v.flatten().tolist()]);
                elif 'left' in right_second_indicator:
                    self.add_dual_edge([left_d_v.flatten().tolist(),
                                        right_second_d_v.flatten().tolist()]);
                else:
                    return None
                '''
                # if np.linalg.norm(right_first_d_v - left_d_v) <= np.linalg.norm(right_second_d_v - left_d_v):
                #     print('linked right_first_d_v')
                #     self.add_dual_edge([left_d_v.flatten().tolist(),
                #                         right_first_d_v.flatten().tolist()]);
                # else:
                #     print('linked right_second_d_v')
                #     self.add_dual_edge([left_d_v.flatten().tolist(),
                #                         right_second_d_v.flatten().tolist()]);

                    
            elif horizontal_cell_left.num_dual_vertex() == 2 and \
                 horizontal_cell_right.num_dual_vertex() == 2:
                print('here h 2 2 ')

                raise ValueError('here h 2 2 ')
                left_first_d_v = horizontal_cell_left.get_first_dual_vertex().flatten().tolist()
                left_first_indicator = horizontal_cell_left.get_first_indicator().flatten().tolist()

                left_second_d_v = horizontal_cell_left.get_second_dual_vertex().flatten().tolist()
                left_second_indicator = horizontal_cell_left.get_second_indicator().flatten().tolist()

                right_first_d_v = horizontal_cell_right.get_first_dual_vertex().flatten().tolist()
                right_second_indicator = horizontal_cell_right.get_second_indicator().flatten().tolist()

                right_second_d_v = horizontal_cell_right.get_second_dual_vertex().flatten().tolist()
                right_first_indicator = horizontal_cell_right.get_first_indicator().flatten().tolist()

                result = self.mc_two_two_connect_h(left_first_d_v, left_first_indicator, left_second_d_v, left_second_indicator, 
                                                    right_first_d_v, right_first_indicator, right_second_d_v, right_second_indicator)
                if result is not None:
                    print('------here h 2 2  edge----')
                    print(result.tolist())
                    assert result.shape == (2, 2), 'result {}'.format(result)
                    self.add_dual_edge(result.tolist());

                '''
                assert not (('right' in left_first_d_v) and ('right' in left_second_d_v))
                assert not (('left' in right_first_d_v) and ('left' in right_second_d_v))

                if 'right' in left_first_indicator:
                    if 'left' in right_first_indicator:
                        self.add_dual_edge([left_first_d_v.flatten().tolist(),
                                            right_first_d_v.flatten().tolist()]);
                    elif 'left' in right_second_indicator:
                        self.add_dual_edge([left_first_d_v.flatten().tolist(),
                                            right_second_d_v.flatten().tolist()]);
                    else:
                        raise ValueError('right connect to left, but right doesn"t connect to left ?')

                elif 'right' in left_second_indicator:
                    if 'left' in right_first_indicator:
                        self.add_dual_edge([left_second_d_v.flatten().tolist(),
                                            right_first_d_v.flatten().tolist()]);
                    elif 'left' in right_second_indicator:
                        self.add_dual_edge([left_second_d_v.flatten().tolist(),
                                            right_second_d_v.flatten().tolist()]);
                    else:
                        raise ValueError('right connect to left, but right doesn"t connect to left ?')
                else:
                    # left not connect to right
                    pass
                '''
                # norms = [norm(left_first_d_v - right_first_d_v),
                #          norm(left_first_d_v - right_second_d_v),
                #          norm(left_second_d_v - right_first_d_v),
                #          norm(left_second_d_v - right_second_d_v)
                #          ]

                # min_index = np.argmin(norms)
                # if min_index == 0:
                #     self.add_dual_edge([left_first_d_v.flatten().tolist(),
                #                         right_first_d_v.flatten().tolist()]);
                # elif min_index == 1:
                #     self.add_dual_edge([left_first_d_v.flatten().tolist(),
                #                         right_second_d_v.flatten().tolist()]);
                # elif min_index == 2:
                #     self.add_dual_edge([left_second_d_v.flatten().tolist(),
                #                         right_first_d_v.flatten().tolist()]);
                # elif min_index == 3:
                #     self.add_dual_edge([left_second_d_v.flatten().tolist(),
                #                         right_second_d_v.flatten().tolist()]);
                # else:
                #     raise ValueError


            else:
                raise ValueError('horizontal_cell_left and horizontal_cell_right has unexpected dual vertex number, \
                                 they are {}, {}'.format(horizontal_cell_left.num_dual_vertex(),
                                                         horizontal_cell_right.num_dual_vertex()))

            # ax1.plot([horizontal_cell_left.get_dual_vertexs().flatten().tolist()[0],
            #           horizontal_cell_right.get_dual_vertexs().flatten().tolist()[0]], 
            #           [horizontal_cell_left.get_dual_vertexs().flatten().tolist()[1], 
            #           horizontal_cell_right.get_dual_vertexs().flatten().tolist()[1]], 'r')

            # return np.array([horizontal_cell_left.get_dual_vertexs().flatten().tolist(),
            #                 horizontal_cell_right.get_dual_vertexs().flatten().tolist()])

        else:
            return None

    def edgeProcV(self, vertical_cell_top, vertical_cell_bottom):

        global ax1

        if vertical_cell_top.is_Root() and vertical_cell_bottom.is_Root():
            self.edgeProcV(vertical_cell_top.cell0, vertical_cell_bottom.cell2)
            self.edgeProcV(vertical_cell_top.cell1, vertical_cell_bottom.cell3)

        elif vertical_cell_top.is_Root() and vertical_cell_bottom.is_Leaf():
            self.edgeProcV(vertical_cell_top.cell0, vertical_cell_bottom)
            self.edgeProcV(vertical_cell_top.cell1, vertical_cell_bottom)

        elif vertical_cell_top.is_Leaf() and vertical_cell_bottom.is_Root():
            self.edgeProcV(vertical_cell_top, vertical_cell_bottom.cell2)
            self.edgeProcV(vertical_cell_top, vertical_cell_bottom.cell3)

        elif vertical_cell_top.is_Leaf() and vertical_cell_bottom.is_Leaf():
            # self.add_dual_edge([vertical_cell_top.get_dual_vertexs()[0].flatten().tolist(),
                            # vertical_cell_bottom.get_dual_vertexs()[0].flatten().tolist()])


            # ax1.plot([vertical_cell_top.get_dual_vertexs().flatten().tolist()[0],
            #           vertical_cell_bottom.get_dual_vertexs().flatten().tolist()[0]], 
            #           [vertical_cell_top.get_dual_vertexs().flatten().tolist()[1], 
            #           vertical_cell_bottom.get_dual_vertexs().flatten().tolist()[1]], 'r')

            # return np.array([vertical_cell_top.get_dual_vertexs().flatten().tolist(),
            #                 vertical_cell_bottom.get_dual_vertexs().flatten().tolist()])

            if vertical_cell_top.num_dual_vertex() == 1 and \
               vertical_cell_bottom.num_dual_vertex() == 1:
                # self.add_dual_edge([vertical_cell_top.get_first_dual_vertex().flatten().tolist(),
                #                     vertical_cell_bottom.get_first_dual_vertex().flatten().tolist()]);


                if 'bottom' in vertical_cell_top.get_first_indicator() and \
                  'top' in vertical_cell_bottom.get_first_indicator():
                    self.add_dual_edge([vertical_cell_top.get_first_dual_vertex().flatten().tolist(),
                                        vertical_cell_bottom.get_first_dual_vertex().flatten().tolist()]);
                else:
                    pass

            elif vertical_cell_top.num_dual_vertex() == 2 and \
                 vertical_cell_bottom.num_dual_vertex() == 1:
                print('here v 2 1 ')

                top_first_d_v = vertical_cell_top.get_first_dual_vertex().flatten().tolist()
                two_first_indicator = vertical_cell_top.get_first_indicator().flatten().tolist()

                top_second_d_v = vertical_cell_top.get_second_dual_vertex().flatten().tolist()
                two_second_indicator = vertical_cell_top.get_second_indicator().flatten().tolist()

                bottom_d_v = vertical_cell_bottom.get_first_dual_vertex().flatten().tolist()
                bottom_indicator = vertical_cell_bottom.get_first_indicator().flatten().tolist()

                result = self.mc_two_one_connect_v(top_first_d_v, two_first_indicator, top_second_d_v, two_second_indicator, 
                                                    bottom_d_v, bottom_indicator)

                if result is not None:
                    assert result.shape == (2, 2), 'result {}'.format(result)
                    self.add_dual_edge(result.tolist());

                '''
                print(top_first_d_v)
                print(top_second_d_v)
                print(bottom_d_v)

                if np.linalg.norm(top_first_d_v - bottom_d_v) <= np.linalg.norm(top_second_d_v - bottom_d_v):
                    print('linked top_first_d_v')

                    self.add_dual_edge([top_first_d_v.flatten().tolist(),
                                        bottom_d_v.flatten().tolist()]);
                else:
                    print('linked top_second_d_v')
                    self.add_dual_edge([top_second_d_v.flatten().tolist(),
                                        bottom_d_v.flatten().tolist()]);
                '''
            elif vertical_cell_top.num_dual_vertex() == 1 and \
                 vertical_cell_bottom.num_dual_vertex() == 2:

                top_d_v = vertical_cell_top.get_first_dual_vertex().flatten().tolist()
                top_indicator = vertical_cell_top.get_first_indicator().flatten().tolist()

                bottom_first_d_v = vertical_cell_bottom.get_first_dual_vertex().flatten().tolist()
                bottom_first_indicator = vertical_cell_bottom.get_first_indicator().flatten().tolist()

                bottom_second_d_v = vertical_cell_bottom.get_second_dual_vertex().flatten().tolist()
                bottom_second_indicator = vertical_cell_bottom.get_second_indicator().flatten().tolist()

                result = self.mc_one_two_connect_v(top_d_v, top_indicator, 
                                                    bottom_first_d_v, bottom_first_indicator, bottom_second_d_v, bottom_second_indicator)

                if result is not None:
                    assert result.shape == (2, 2), 'result {}'.format(result)
                    self.add_dual_edge(result.tolist());


                '''
                print('here v 1 2')
                top_d_v = vertical_cell_top.get_first_dual_vertex()
                bottom_first_d_v = vertical_cell_bottom.get_first_dual_vertex()
                bottom_second_d_v = vertical_cell_bottom.get_second_dual_vertex()

                print(top_d_v)
                print(bottom_first_d_v)
                print(bottom_second_d_v)

                if np.linalg.norm(bottom_first_d_v - top_d_v) <= np.linalg.norm(bottom_second_d_v - top_d_v):
                    print('linked bottom_first_d_v')
                    self.add_dual_edge([top_d_v.flatten().tolist(),
                                        bottom_first_d_v.flatten().tolist()]);
                else:
                    print('linked bottom_second_d_v')
                    self.add_dual_edge([top_d_v.flatten().tolist(),
                                        bottom_second_d_v.flatten().tolist()]);
                '''
                    
            elif vertical_cell_top.num_dual_vertex() == 2 and \
                 vertical_cell_bottom.num_dual_vertex() == 2:

                raise ValueError('here v 2 2')


                top_first_d_v = vertical_cell_top.get_first_dual_vertex().flatten().tolist()
                two_first_indicator = vertical_cell_top.get_first_indicator().flatten().tolist()

                top_second_d_v = vertical_cell_top.get_second_dual_vertex().flatten().tolist()
                two_second_indicator = vertical_cell_top.get_second_indicator().flatten().tolist()

                bottom_first_d_v = vertical_cell_bottom.get_first_dual_vertex().flatten().tolist()
                bottom_first_indicator = vertical_cell_bottom.get_first_indicator().flatten().tolist()

                bottom_second_d_v = vertical_cell_bottom.get_second_dual_vertex().flatten().tolist()
                bottom_second_indicator = vertical_cell_bottom.get_second_indicator().flatten().tolist()

                result = self.mc_two_two_connect_v(top_first_d_v, two_first_indicator, top_second_d_v, two_second_indicator, 
                                                    bottom_first_d_v, bottom_first_indicator, bottom_second_d_v, bottom_second_indicator)



                if result is not None:
                    assert result.shape == (2, 2), 'result {}'.format(result)
                    self.add_dual_edge(result.tolist());

                '''
                print('here v 2 2 ')

                top_first_d_v = vertical_cell_top.get_first_dual_vertex()
                top_second_d_v = vertical_cell_top.get_second_dual_vertex()

                bottom_first_d_v = vertical_cell_bottom.get_first_dual_vertex()
                bottom_second_d_v = vertical_cell_bottom.get_second_dual_vertex()

                norms = [norm(top_first_d_v - bottom_first_d_v),
                         norm(top_first_d_v - bottom_first_d_v),
                         norm(top_second_d_v - bottom_first_d_v),
                         norm(top_second_d_v - bottom_second_d_v)
                         ]

                min_index = np.argmin(norms)
                if min_index == 0:
                    self.add_dual_edge([top_first_d_v.flatten().tolist(),
                                        bottom_first_d_v.flatten().tolist()]);
                elif min_index == 1:
                    self.add_dual_edge([top_first_d_v.flatten().tolist(),
                                        bottom_first_d_v.flatten().tolist()]);
                elif min_index == 2:
                    self.add_dual_edge([top_second_d_v.flatten().tolist(),
                                        bottom_first_d_v.flatten().tolist()]);
                elif min_index == 3:
                    self.add_dual_edge([top_second_d_v.flatten().tolist(),
                                        bottom_second_d_v.flatten().tolist()]);
                else:
                    raise ValueError
                '''

        else:
            return None


    def faceProc(self):
        if self.is_Leaf():
            return None
        elif self.is_Root():


            self.cell0.faceProc()
            self.cell1.faceProc()
            self.cell2.faceProc()
            self.cell3.faceProc()

            self.edgeProcH(self.cell0, self.cell1)
            self.edgeProcH(self.cell2, self.cell3)

            self.edgeProcV(self.cell2, self.cell0)
            self.edgeProcV(self.cell3, self.cell1)

        else:
            pass

    def add_dual_vertex(self, dual_vertex):
        # can be more than one in a Leaf
        assert self.is_Leaf()
        assert not hasattr(self, 'dual_vertex')
        try:
            self.dual_vertexs.append(dual_vertex)
        except AttributeError:
            self.dual_vertexs = [dual_vertex]

    def get_first_dual_vertex(self):
        return self.get_dual_vertexs()[0]

    def get_second_dual_vertex(self):
        assert self.num_dual_vertex() == 2
        return self.get_dual_vertexs()[1]

    def get_dual_vertexs(self):
        assert self.is_Leaf()
        assert len(self.dual_vertexs) <= 2
        return self.dual_vertexs

    def num_dual_vertex(self):
        assert self.is_Leaf()
        return len(self.get_dual_vertexs())

    def get_all_dual_vertex(self, list_to_add):
        if self.is_Leaf():
            for each_dual_vertex in self.get_dual_vertexs():
                list_to_add.append(each_dual_vertex)
        elif self.is_Root():
            self.cell0.get_all_dual_vertex(list_to_add)
            self.cell1.get_all_dual_vertex(list_to_add)
            self.cell2.get_all_dual_vertex(list_to_add)
            self.cell3.get_all_dual_vertex(list_to_add)
        else:
            pass

    def is_point_in_cell(self, two_d_point):
        x = two_d_point[0][0]
        y = two_d_point[1][0]

        if (self.xmin <= x) and (x <= self.xmax) and \
            (self.ymin <= y) and (y <= self.ymax):
            return True
        else:
            return False


    def dual_contouring(self, implicit_object_instance, intersection_points):


        def intersection_of_two_paramatric_line(dx_0, dy_0, x_0, y_0, 
                                                dx_1, dy_1, x_1, y_1):
            
            epsilon = 0.1
            if abs(dx_0) <= epsilon and dx_1 != 0:
                return np.array([[x_0], [(dy_1/dx_1)*x_0 - (dy_1/dx_1) * x_1 + y_1]])
            elif abs(dx_1) <= epsilon and dx_0 != 0:
                return np.array([[x_1], [((dy_0/dx_0)*x_1 - (dy_0/dx_0) * x_0 + y_0)]])
            elif abs(dx_1) <= epsilon and abs(dx_0) <= epsilon:
                if abs(x_0 - x_1) <= epsilon:
                    answer = np.array([[x_0], [(y_0+y_1)/2]])
                else:
                    raise ValueError('No answer')
            elif abs(dy_0) <= epsilon and dy_1 != 0:
                return np.array([[(y_0 - y_1 + (dy_1/dx_1)*x_1)/(dy_1/dx_1)], [y_0]])
            elif abs(dy_1) <= epsilon and dy_0 != 0:
                return np.array([[(y_1 - y_0 + (dy_0/dx_0)*x_0)/(dy_0/dx_0)], [y_1]])
            elif abs(dy_1) <= epsilon and abs(dy_0) <= epsilon:
                if abs(y_0 - y_1) <= epsilon:
                    answer = np.array([[(x_0+x_1)/2], [y_0]])
                else:
                    raise ValueError('No answer')
            elif abs(dx_0 - dx_1) <= 0.1 and abs(dy_0 - dy_1) <= epsilon:
                print('two tangent are near parallel')
                answer = np.array([[(x_0+x_1)/2], [(y_0+y_1)/2]])
            else:
                # Ax = b 
                A = np.array([[dy_0/dx_0, -1],
                                   [dy_1/dx_1, -1]], dtype='float')

                b = np.array([[(dy_0/dx_0)*x_0 - y_0],
                              [(dy_1/dx_1)*x_1 - y_1]])

                answer = np.dot(np.linalg.inv(A), b)

            assert isinstance(answer, np.ndarray)
            return answer

        if self.is_Root():
            self.cell0.dual_contouring(implicit_object_instance, intersection_points)
            self.cell1.dual_contouring(implicit_object_instance, intersection_points)
            self.cell2.dual_contouring(implicit_object_instance, intersection_points)
            self.cell3.dual_contouring(implicit_object_instance, intersection_points)

        elif self.is_Leaf():

            for each_pair_edge in self.edge_vectices:
                point_0_x = each_pair_edge[0][0]
                point_0_y = each_pair_edge[0][1]

                point_1_x = each_pair_edge[1][0]
                point_1_y = each_pair_edge[1][1]


                point_0_derivative = implicit_object_instance.derivative_at_point(np.array([[point_0_x], [point_0_y]]))
                point_1_derivative = implicit_object_instance.derivative_at_point(np.array([[point_1_x], [point_1_y]]))

                print('answer of two dot product')
                print(point_0_derivative/np.linalg.norm(point_0_derivative.T))
                print(point_1_derivative/np.linalg.norm(point_1_derivative))
                cos_theta = np.dot(point_0_derivative.T/np.linalg.norm(point_0_derivative.T),
                                   point_1_derivative/np.linalg.norm(point_1_derivative))

                print(cos_theta)


                if abs(cos_theta) >= 0.9:
                    print('no sharp edge, return mid point, dodgy')
                    self.add_dual_vertex(np.array([[point_0_x/2 + point_1_x/2], [point_1_y/2 + point_1_y/2]]))
                    continue
                else:
                    point_0_dx = point_0_derivative[0][0]
                    point_0_dy = point_0_derivative[1][0]

                    rotated_point_0_dx = -1 * point_0_dy
                    rotated_point_0_dy = point_0_dx

                    point_1_dx = point_1_derivative[0][0]
                    point_1_dy = point_1_derivative[1][0]

                    rotated_point_1_dx = -1 * point_1_dy
                    rotated_point_1_dy = point_1_dx

                    intersection_point = intersection_of_two_paramatric_line(
                        rotated_point_0_dx, rotated_point_0_dy, point_0_x, point_0_y,
                        rotated_point_1_dx, rotated_point_1_dy, point_1_x, point_1_y)


                    # self.add_dual_vertex(intersection_point)

                    # assert self.is_point_in_cell(intersection_point), \
                    #     'dual vertex {} not in Cell {}'.format(intersection_point, self.xmin_xmax_ymin_ymax())

                    
                    if self.is_point_in_cell(intersection_point):
                        self.add_dual_vertex(intersection_point)
                        intersection_points.append(intersection_point)
                    elif not self.is_point_in_cell(intersection_point):
                        print('dodgy not in cell return origin value')
                        self.add_dual_vertex(np.array([[point_0_x/2 + point_1_x/2], [point_1_y/2 + point_1_y/2]]))
                        intersection_points.append(np.array([[point_0_x/2 + point_1_x/2], [point_1_y/2 + point_1_y/2]]))

                    else:
                        raise ValueError
                    

        else:
            pass

    # TODO: a function to loop through all subcell if cell is a root

    def get_all_dual_edge(self, edges):

        assert isinstance(edges, list)

        if self.is_Root():
            if self.has_dual_edge():
                edges.append(self.get_dual_edge())
            else:
                pass

            self.cell0.get_all_dual_edge(edges)
            self.cell1.get_all_dual_edge(edges)
            self.cell2.get_all_dual_edge(edges)
            self.cell3.get_all_dual_edge(edges)
        else:
            pass


    # def get_all_dual_vertice(self, vertices):

    #     assert isinstance(edges, list)

    #     if self.is_Root():
    #         if self.has_dual_edge():
    #             edges.append(self.get_dual_edge())
    #         else:
    #             pass

    #         self.cell0.get_all_dual_edge(edges)
    #         self.cell1.get_all_dual_edge(edges)
    #         self.cell2.get_all_dual_edge(edges)
    #         self.cell3.get_all_dual_edge(edges)
    #     else:
    #         pass




# class Root(Cell):
#     pass

# class Full(Cell):
#     pass

# class Empty(Cell):
#     pass

# class Leaf(Cell):
#     pass


# In[167]:

def build_tree(cell, counter):
    assert isinstance(cell, Cell)
    assert counter >= 0
    if counter == 0:            
        pass
    else:
        
        counter -= 1
        cell.to_Root()
        build_tree(cell.cell0, counter)
        build_tree(cell.cell1, counter)
        build_tree(cell.cell2, counter)
        build_tree(cell.cell3, counter)

def collapse(cell, implicit_object):
    
    assert isinstance(cell, Cell)
    assert isinstance(implicit_object, ImplicitObject)

    if cell.is_Root():
        
        if cell.cell0.is_Root():
            collapse(cell.cell0, implicit_object)
        else:
            assert not cell.cell0.is_NotInitialized(), \
                'please initialize type i.e. call eval_type() before collapse'
            pass
        
        if cell.cell1.is_Root():
            collapse(cell.cell1, implicit_object)
        else:
            assert not cell.cell1.is_NotInitialized(), \
                'please initialize type i.e. call eval_type() before collapse'
            pass

        if cell.cell2.is_Root():
            collapse(cell.cell2, implicit_object)
        else:
            assert not cell.cell2.is_NotInitialized(), \
                'please initialize type i.e. call eval_type() before collapse'
            pass
            
        if cell.cell3.is_Root():
            collapse(cell.cell3, implicit_object)
        else:
            assert not cell.cell3.is_NotInitialized(), \
                'please initialize type i.e. call eval_type() before collapse'
            pass

        if ((cell.cell0.is_Full()) & 
           (cell.cell1.is_Full()) &
           (cell.cell2.is_Full()) &
           (cell.cell3.is_Full())) :
            
            cell.to_Full()
        
        elif ((cell.cell0.is_Empty()) & 
             (cell.cell1.is_Empty()) &
             (cell.cell2.is_Empty()) &
             (cell.cell3.is_Empty())):
            
            cell.to_Empty()
        
        else:
            pass


def implicit_3d_plot():
    def plot_implicit(fn, bbox=(-1,1)):
        ''' create a plot of an implicit function
        fn  ...implicit function (plot where fn==0)
        bbox ..the x,y,and z limits of plotted interval'''
        xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        A = np.linspace(xmin, xmax, 100) # resolution of the contour
        B = np.linspace(xmin, xmax, 15) # number of slices
        A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

        for z in B: # plot contours in the XY plane
            X,Y = A1,A2
            Z = fn(X,Y,z)
            cset = ax.contour(X, Y, Z+z, [z], zdir='z')
            # [z] defines the only level to plot for this contour for this value of z

        for y in B: # plot contours in the XZ plane
            X,Z = A1,A2
            Y = fn(X,y,Z)
            cset = ax.contour(X, Y+y, Z, [y], zdir='r')

        for x in B: # plot contours in the YZ plane
            Y,Z = A1,A2
            X = fn(x,Y,Z)
            cset = ax.contour(X+x, Y, Z, [x], zdir='x')

        # must set plot limits because the contour will likely extend
        # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
        # to encompass all values in the contour.
        ax.set_zlim3d(zmin,zmax)
        ax.set_xlim3d(xmin,xmax)
        ax.set_ylim3d(ymin,ymax)

        plt.show()


    # In[2]:

    # def goursat_tangle(x,y,z):
    #     a,b,c = 0.0,-5.0,11.8
    #     return x**4+y**4+z**4+a*(x**2+y**2+z**2)**2+b*(x**2+y**2+z**2)+c

    # def square_minus_ball(x,y,z):
    #     return (1/x)**2 + (1/y)**2 + (1/z)**2 - 2

    def square_minus_ball(x,y,z):
        return (abs(x*y*z) < 0.05) * ((0.9 - np.sqrt(x*x + y*y + z*z))**2 < 0.01)


    plot_implicit(square_minus_ball)


def implicit_object_tests():
    # In[5]:

    neg_one_zero = np.array([[-1], [0]])

    zero_zero = np.array([[0],
                          [0]])

    zero_point_five_zero_point_five = np.array([[0.5],
                          [0.5]])

    zero_one = np.array([[0],
                         [1]])

    zero_two = np.array([[0],
                         [2]])

    zero_neg_one = np.array([[0],
                         [-1]])

    zero_neg_two = np.array([[0],
                         [-2]])

    one_zero = np.array([[1],
                         [0]])

    two_zero = np.array([[2],
                         [0]])

    two_two = np.array([[2],
                          [2]])


    # In[6]:

    # radius one center at origin cricle
    circle = ImplicitObject(lambda x, y: np.sqrt((x)**2 + (y)**2) - 1)
    print(circle.eval_point(zero_zero)) # -1.0
    print(circle.eval_point(zero_one)) # 0.0
    print(circle.eval_point(two_two)) # 1.82842712475

    negation_circle = circle.negate()
    print(negation_circle.eval_point(zero_zero)) # 1.0
    print(negation_circle.eval_point(zero_one)) # 0.0
    print(negation_circle.eval_point(two_two)) # -1.82842712475


    # In[7]:

    # left x0 plane lambda x, _:x - x0
    left_1 = ImplicitObject(lambda x, _: x - 1);
    print(left_1.eval_point(zero_zero)) # -1
    print(left_1.eval_point(one_zero)) # 0
    print(left_1.eval_point(two_zero))  # 1


    # In[8]:

    # right x0 plane lambda x, _:x0 - x
    right_1 = ImplicitObject(lambda x, _:1 - x);
    print(right_1.eval_point(zero_zero)) # 1 
    print(right_1.eval_point(one_zero)) # 0
    print(right_1.eval_point(two_zero)) # -1


    # In[9]:

    # bottom y0 plane lamda _, y: y0 - y
    top_1 = ImplicitObject(lambda _, y: 1 - y);
    print(top_1.eval_point(zero_zero)) # 1
    print(top_1.eval_point(zero_one)) # 0
    print(top_1.eval_point(zero_two)) # -1


    # In[10]:

    # top y0 plane lambda _, y:y + (- y0)
    bottom_1 = ImplicitObject(lambda _, y: y +  - 1);
    print(bottom_1.eval_point(zero_zero)) # -1
    print(bottom_1.eval_point(zero_one)) # 0
    print(bottom_1.eval_point(zero_two)) # 1


    # In[11]:

    circle_union_bottom = circle.union(bottom_1)
    print(circle_union_bottom.eval_point(zero_zero))
    print(circle_union_bottom.eval_point(zero_one))
    print(circle_union_bottom.eval_point(zero_neg_two))
    print(circle_union_bottom.eval_point(zero_two))
    print(circle_union_bottom.eval_point(two_zero))



    # In[13]:

    circle = ImplicitCircle(0, 0, 1)
    circle.visualize(-2, 2, -2, 2, 100)


    # In[14]:

    circle = ImplicitCircle(0,0,1)
    print(circle.eval_point(zero_zero)) # -1.0
    print(circle.eval_point(zero_one)) # 0.0
    print(circle.eval_point(two_two)) # 1.82842712475

    negation_circle = circle.negate()
    print(negation_circle.eval_point(zero_zero)) # 1.0
    print(negation_circle.eval_point(zero_one)) # 0.0
    print(negation_circle.eval_point(two_two)) # -1.82842712475

# In[16]:

    left_1 = Left(1)
    left_1.visualize(-3, 3, -3, 3, 100)

    right_1 = Right(1)
    right_1.visualize(-3, 3, -3, 3, 100)


    Lower_1 = Lower(1)
    Lower_1.visualize(-3, 3, -3, 3, 100)


    Upper_1 = Upper(1)
    Upper_1.visualize(-3, 3, -3, 3, 100)



    # In[18]:

    rect = ImplicitRectangle(0, 1, 0, 1)
    print(rect.eval_point(zero_zero))
    print(rect.eval_point(zero_one))
    print(rect.eval_point(zero_point_five_zero_point_five))
    print(rect.eval_point(two_zero))
    rect.visualize(-1, 1, -1, 1, 100)

def derivative_test():
    circle = ImplicitCircle(0,0,1)
    print(circle.derivative_at_point(np.array([[1], [1]]))) # np.array([[0.70710678], [0.70710678]])

    rectangle = ImplicitRectangle(0,1,0,1)
    print(rectangle.derivative_at_point(np.array([[0.9], [1]])))

# In[162]:
def main():

    from mpl_toolkits.mplot3d import axes3d
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    h = ImplicitRectangle(0.1, 0.25, 0.1, 0.9).union(ImplicitRectangle(0.1, 0.6, 0.1, 0.35)).union(ImplicitCircle(0.35, 0.35, 0.25)).intersect(
            (ImplicitCircle(0.35, 0.35, 0.1).union(ImplicitRectangle(0.25, 0.45, 0.1, 0.35))).negate()
        )

    i = ImplicitRectangle(0.75, 0.9, 0.1, 0.55).union(ImplicitCircle(0.825, 0.75, 0.1))

    hi = h.union(i)


    # hi = ImplicitStar(0.2, 0.1, 10, 0.5, 0.5)
    # hi = ImplicitTree()


    rect_0 = ImplicitRectangle(0.2, 0.6, 0.2, 0.6)
    hi = rect_0.displace(200, 10)


    # hi.visualize_bitmap(0, 1, 0, 1, 500)
    # hi.visualize_distance_field(0, 1, 0, 1, 500)

    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111, aspect='equal')
    # ax1.set_xlim([-0.5, 1.5])
    # ax1.set_ylim([-0.5, 1.5])

    c = Cell(0, 1, 0, 1, 'NotInitialized')
    build_tree(c, 7)
    c.initialise_cell_type(hi)
    # c.visualize(ax1)
    c.print_type()

    c.check_not_initialized_exists()
    c.sampled_distance_fields(hi)

    # c.visualize(ax1)

    c.check_not_initialized_exists()
    collapse(c, hi)

    c.visualize(ax1)

    edges = []
    edges = c.marching_cube(hi, edges)

    print('length of edges')
    print(len(edges))
    print(edges)

    intersection_points = []
    c.dual_contouring(hi, intersection_points)
    c.faceProc()


    intersection_points = [] 
    c.get_all_dual_vertex(intersection_points)


    dual_edges = []
    c.get_all_dual_edge(dual_edges)
    print('len of dual_edges')
    print(len(dual_edges))
    assert(len(dual_edges) > 0)
    # print('---edge---')
    # print(intersection_points) # weird format

    # import numpy as np
    # import pylab as pl
    from matplotlib import collections  as mc

    lc = mc.LineCollection(edges, linewidths=2)
    for dual_edge in dual_edges:
        lc = mc.LineCollection(dual_edge, linewidths=2, color='red')
        ax1.add_collection(lc)


    for edge, intersect_point in zip(edges, intersection_points):
        edges_mid_point_x = (edge[0][0] + edge[1][0])/2
        edges_mid_point_y = (edge[0][1] + edge[1][1])/2

        # ax1.plot([intersect_point[0][0]], [intersect_point[1][0]], 'ro')
        # marching cube mid point
        # ax1.plot([edges_mid_point_x], [edges_mid_point_y], 'go')


        # marching cube mid point to dual vertex..
        # ax1.plot([edges_mid_point_x, intersect_point[0][0]], [edges_mid_point_y, intersect_point[1][0]], 'yellow')


        # marching cube
        # ax1.plot([edge[0][0]], [edge[0][1]], 'go')
        # ax1.plot([edge[1][0]], [edge[1][1]], 'go')
        # ax1.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'green')



    # ax.autoscale()
    # ax.margins(0.1)

    plt.show()

def smooth_union_demo():
    import numpy as np
    import matplotlib.pyplot as plt    

    rect_0 = ImplicitRectangle(0.2, 0.6, 0.2, 0.6)

    rect_1 = ImplicitRectangle(0.4, 0.8, 0.4, 0.8)
    
    # rect_new = rect_0.polynomial_smooth_union(rect_1)
    # # rect_new.visualize_distance_field(0, 1, 0, 1)
    # rect_new.visualize_bitmap(0, 1, 0, 1)

    # rect_new = rect_0.union(rect_1)
    # rect_new.visualize_bitmap(0, 1, 0, 1)

    # rect_new = rect_0.exponential_smooth_union(rect_1)
    # rect_new.visualize_distance_field(0, 1, 0, 1)

    rect_new = rect_0.displace(200, 10)
    rect_new.visualize_bitmap(0, 1, 0, 1)
    plt.show()


if __name__ == '__main__':
    # smooth_union_demo()

    main()
