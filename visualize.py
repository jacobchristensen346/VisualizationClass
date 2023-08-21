import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from matplotlib import cm
import math
import sys

class Visualize:
    """
    allows for a variety of methods to visualize the calculated 3Dmpdf. Requires both the 3Dmpdf
    array and the 3 arrays x, y, z from which the grid was created.
        
    """
        
    def __init__(self, m, x, y, z):
        """
        Args:
        
            m: 3Dmpdf, as a 3D numpy array
            x, y, z: each are numpy arrays whose values are used to create the meshgrid for 3Dmpdf
            
        """
        
        if not all(isinstance(i, np.ndarray) for i in [m, x, y, z]):
            raise ValueError('Please enter numpy arrays for each argument')
            
        self.m = m
        self.x = x
        self.y = y
        self.z = z
        self.a = None #these two instance attributes will construct an arbitrary grid,
        self.b = None #defined here to allow for cross function usage
        
    def three_points(self, p1, p2, p3):
        #find normal from three points given
        #find two vectors from the three points which lie on the desired plane
        vec1 = p2 - p1
        vec2 = p3 - p1
        #now cross these two vectors to find a vector normal to the plane
        normal = np.cross(vec1, vec2)
    
        #now calculate the centroid of the three points given
        x_pos = (p1[0] + p2[0] + p3[0]) / 3
        y_pos = (p1[1] + p2[1] + p3[1]) / 3
        z_pos = (p1[2] + p2[2] + p3[2]) / 3
        cen_pt = np.array([x_pos, y_pos, z_pos])
        print('Center Point:', cen_pt)
        
        return normal, cen_pt
    
    
    def make_slice(self, len_a=None, len_b=None, dr=None, use_norm=None, cen_pt=None, normal=None, 
                   p1=None, p2=None, p3=None, inc_pts=None):
        """
        
        Args:
        
            len_a/len_b: the side length of the rectangular slice to be taken through the data
            dr: determines the spacing of the grid (if dr=0.5, then there are 2 measurements every angstrom)
            use_norm: When True, will create slice from user given normal vector and center point. When
                false, will create slice from three points given by the user
            cen_pt: The center of the desired slice. Used when use_norm is True
            normal: The normal vector to desired plane. Used when use_norm is True
            p1, p2, p3: Three points in 3D space given as numpy arrays. The desired plane goes through these points.
                Used when use_norm is False
            inc_pts: If True, will ensure that points p1, p2, p3 that specifiy the desired plane will be
                            included in the slice that is plotted
            
        Returns:
        
            Returns 2D array, representing slice through 3Dmpdf

        """
        
        if dr is None:
            dr = 1
        if use_norm is None:
            use_norm = True
        if len_a is None:
            len_a = 10
        if len_b is None:
            len_b = 10
        if cen_pt is None:
            cen_pt = np.array([0, 0, 0])
        if normal is None:
            normal = np.array([1, 0, 0])
        if p1 is None:
            p1 = np.array([0, 1, 0])
        if p2 is None:
            p2 = np.array([1, 0, 0])
        if p3 is None:
            p3 = np.array([0, 0, 1])
        if inc_pts is None:
            inc_pts = True
        
        #First check if use_norm is False. If so, access three_points function 
        #to calculate the normal and cen_pt of the desired plane
        if use_norm is False:
            normal, cen_pt = self.three_points(p1, p2, p3)
        
        #here is a check to alert the user if the given side length goes out of bounds of the 3Dmpdf
        if ((cen_pt[0] + side_len / 2 <= max(self.x) and cen_pt[0] - side_len / 2 >= 
             min(self.x)) and (cen_pt[1] + side_len / 2 <= max(self.y) and cen_pt[1] - side_len / 2 >=
                 min(self.y)) and (cen_pt[2] + side_len / 2 <= max(self.z) and cen_pt[2] - side_len / 2 >= 
                     min(self.z))) == False:
            raise ValueError('Given side_len and cen_pt go out of bounds of 3Dmpdf grid')
         
        #ensure that our basis vector v1 is not the same as normal
        v1 = np.array([1, 0, 0])
        if np.allclose(v1, normal):
            v1 = np.array([0, 1, 0])
    
        #now make a matrix which will reflect any vector onto the orthogonal
        #complement of the normal vec, which is our desired plane
        #This is done by subtracting from the vector its component along the normal vector
        m_norm = np.eye(3) - (np.outer(normal, normal.T) / normal.T.dot(normal))
        
        #now reflect v1 using m_norm
        v1 = m_norm.dot(v1)
        #and create a new vector v2 that is orthogonal to both v1 and normal
        v2 = np.cross(normal, v1)
        #we now have 2 vectors to form our plane
    
        #now create and normalize Q, which will rotate an arbitrary 
        #slice to the orientation we desire
        Q = np.column_stack((v1, v2, np.zeros_like(v1)))
        Q[:,:2] /= np.linalg.norm(Q[:,:2], axis = 0)

        #Check if inc_pts is true. If so, ensure that side length
        #will include p1, p2, and p3 in the slice
        if inc_pts == True:
            dist1 = np.linalg.norm(p1 - cen_pt)
            dist2 = np.linalg.norm(p2 - cen_pt)
            dist3 = np.linalg.norm(p3 - cen_pt)
            if (side_len/2 < dist1 or side_len/2 < dist2 or side_len/2 < dist3):
                side_len = 2 * max([dist1, dist2, dist3])
                print('Adjusted side_len to include 3 points specified. New side_len:', side_len)
    
        #now create an arbitrary slice
        self.a = np.arange(-len_a / 2, len_a / 2, dr)
        self.b = np.arange(-len_b / 2, len_b / 2, dr)
        self.a = np.append(self.a, len_a / 2)
        self.b = np.append(self.b, len_b / 2)
        A,B = np.meshgrid(self.a, self.b)
        locations = np.array([A.reshape(-1), B.reshape(-1), np.zeros(A.size)]) #the slice starts on the x-y plane
        #now move locations onto our two vectors, and add cen_pt to move slice into position
        locations = Q.dot(locations).T + (cen_pt)
    
        #now we need to interpolate our 3Dmpdf function over this slice
        points = (self.x, self.y, self.z)
        interp = interpn(points, self.m, locations) #list of values of 3Dmpdf at locations
        slice1 = interp.reshape(len(self.a),len(self.b))
        
        return slice1
    
    
    def convert_1D(self):
        
        """
        takes XX,YY,ZZ, defined from meshgrid(x,y,z), and converts the meshgrid to 1D
        representing distances from origin r.
        
        Returns:
        
            Flattened 3Dmpdf array along with the corresponding distance from origin array rarray1
        
        """
            
        #dont include plotting function, return r array and mpdf
        XX, YY, ZZ = np.meshgrid(self.x, self.y, self.z)
        rarray=np.sqrt(XX**2 + YY**2 + ZZ**2) #make array of distances from origin r for each x,y,z point
        rarray1=np.ravel(rarray) #put into form to plot
        m1=np.ravel(self.m)
        
        return rarray1, m1
    
    
    def plot_slice(self, sliced_plane, contour=None, smart_scale=None, cmin=None, cmax=None, levels=None):
    
        """
        Args:
        
            sliced_plane: slice returned by "make_slice" function
            cmin, cmax: Sets the values to which the maximum and minimum color values will be assigned. If no value is
                given for both cmin and cmax, default scaling will be used by matplotlib
            contour: If True, will plot the calculated slice as a contour plot. 
                If False, as a continuous surface
            smart_scale: If True, will scale the colormap for the user by setting cmin and cmax to the range
                of 2 standards deviations (intended to eliminate outliers)
            levels: designates the number of levels the contour plot should have. Required if user designates
                cmin and cmax while contour == True
        
        Returns:
    
            Plots the 2D slice through 3Dmpdf as desired
        
        """
        if contour is None:
            contour = False
        if smart_scale is None:
            smart_scale = False
        
        if smart_scale == True:
            data = np.ravel(sliced_plane)
            n = len(data)
            mean = sum(data) / n
            std_dev = np.sqrt(sum((x - mean) ** 2 for x in data) / n)
            cmin = mean - 2 * std_dev
            cmax = mean + 2 * std_dev
        
        #now we plot either a contour or continuous
        if contour == True:
            
            if cmin != None and cmax != None:
                plt.contourf(self.a, self.b, sliced_plane, levels = np.linspace(cmin, cmax, levels), 
                             extend = 'both' , cmap = cm.magma)
                plt.colorbar()
            else:
                plt.contourf(self.a, self.b, sliced_plane, cmap = cm.magma)
                plt.colorbar()
            
        else:
            plt.imshow(sliced_plane, cmap = cm.magma, origin = 'lower', 
                       extent = [min(self.a), max(self.a), min(self.a), max(self.a)])
            if cmin != None and cmax != None:
                plt.clim(cmin, cmax)
            plt.colorbar()
