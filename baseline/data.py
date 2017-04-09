from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from kitti_data.draw import *
from kitti_data.io import *

from net.utility.draw import *
from net.processing.boxes3d import *


from mpl_toolkits.mplot3d import Axes3D


# run functions --------------------------------------------------------------------------



def lidar_to_front(points,
                            v_res=0.42,
                            h_res = 0.35,
                            v_fov = (-24.9, 2.0),
                            d_range = (0,100),
                            y_fudge=3
                            ):
    """ Takes point cloud data as input and creates a 360 degree panoramic
        image, returned as a numpy array.

    Args:
        points: (np array)
            The numpy array containing the point cloud. .
            The shape should be at least Nx3 (allowing for more columns)
            - Where N is the number of points, and
            - each point is specified by at least 3 values (x, y, z)
        v_res: (float)
            vertical angular resolution in degrees. This will influence the
            height of the output image.
        h_res: (float)
            horizontal angular resolution in degrees. This will influence
            the width of the output image.
        v_fov: (tuple of two floats)
            Field of view in degrees (-min_negative_angle, max_positive_angle)
        d_range: (tuple of two floats) (default = (0,100))
            Used for clipping distance values to be within a min and max range.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical image height do not match the actual data.
    Returns:
        A numpy array representing a 360 degree panoramic image of the point
        cloud.
    """
    # Projecting to 2D
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    r_points = points[:, 3]
    d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    #d_points = np.sqrt(x_points**2 + y_points**2 + z_points**2) # abs distance

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_fov_total = -v_fov[0] + v_fov[1]

    # CONVERT TO RADIANS
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = -(np.arctan2(z_points, d_points) / v_res_rad)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total/v_res) / (v_fov_total* (np.pi / 180))
    h_below = d_plane * np.tan(-v_fov[0]* (np.pi / 180))
    h_above = d_plane * np.tan(v_fov[1] * (np.pi / 180))
    y_max = int(np.ceil(h_below+h_above + y_fudge))

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(360.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)

    # CLIP DISTANCES
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])

    # CONVERT TO IMAGE ARRAY
    img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)
    img[y_img, x_img] = scale_to_255(d_points, min=d_range[0], max=d_range[1])

    return img

## objs to gt boxes ##
def obj_to_gt_boxes3d(objs):

    num        = len(objs)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels  = np.zeros((num),    dtype=np.int32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        label = 1 #<todo>

        gt_labels [n]=label
        gt_boxes3d[n]=b

    return  gt_boxes3d, gt_labels


## lidar to top ##
def lidar_to_top(lidar,objects):
    if 0: # crop the field of view
        crop=20

        # TOP_X_MIN=-crop
        # TOP_X_MAX=crop
        # TOP_Y_MIN=-crop
        # TOP_Y_MAX=crop

    lidar=lidar[lidar[:,0]<TOP_X_MAX]
    lidar=lidar[lidar[:,1]<TOP_Y_MAX]
    lidar=lidar[lidar[:,0]>TOP_X_MIN]
    lidar=lidar[lidar[:,1]>TOP_Y_MIN]


    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]
    '''
    print(pxs.max(),pxs.min())
    print(pys.max(),pys.min())
    print(pzs.max(),pzs.min())'''

    def find_highest_and_density_in_cell(pxs,pys,pzs,resolution):
        idx_max=np.zeros((resolution,resolution),dtype=int)
        density=np.zeros((resolution,resolution))
        for i in range(0,pxs.size):
            cell=(math.floor((pxs[i]-TOP_X_MIN)/x_scale),math.floor((pys[i]-TOP_Y_MIN)/y_scale))
            if density[cell]==0 or pzs[i]>pzs[idx_max[cell]]:
                idx_max[cell]=i
            density[cell]+=1
        return idx_max,density

    # pixels size
    resolution=448
    x_scale=(TOP_X_MAX-TOP_X_MIN)/resolution
    y_scale=(TOP_Y_MAX-TOP_Y_MIN)/resolution

    # find features
    idx_max, density=find_highest_and_density_in_cell(pxs,pys,pzs,resolution)

    has_points=density!=0
    intensity=np.zeros((resolution,resolution))
    height=np.zeros((resolution,resolution))
    intensity[has_points]=prs[idx_max][has_points]
    height[has_points]=pzs[idx_max][has_points]-TOP_Z_MIN

    # normalize density
    density=np.log(density+1)/math.log(64)
    density[density<1]=1

    image=np.r_[density, intensity, height]

    def get_boxes(objects,x_scale, y_scale):
        boxes=[]
        for obj in objects:
            box = obj.box[0:4, 0:2]
            box = np.r_[box, box[0:1, :]]
            box = (box - np.array([TOP_X_MIN, TOP_Y_MIN]).T) / np.array([x_scale, y_scale]).T
            box = box.T
            boxes[n]=np.c_[box[1],box[0]]
        return boxes

    boxes=get_boxes(objects,x_scale, y_scale)


    if 0: # plot bird views
        def plot_boxes():
            for obj in objects:
                box=obj.box[0:4,0:2]
                box=np.r_[box,box[0:1,:]]
                box=(box-np.array([TOP_X_MIN,TOP_Y_MIN]).T)/np.array([x_scale,y_scale]).T
                box=box.T
                plt.plot(box[1],box[0]) # imshow and scatter uses different axes
                #plt.plot(box[0],box[1])

        fig=plt.figure()
        a=fig.add_subplot(2,2,1)
        plt.imshow(density)
        plot_boxes()
        a=fig.add_subplot(2,2,2)
        plt.imshow(intensity)
        plot_boxes()
        a=fig.add_subplot(2,2,3)
        plt.imshow(height)
        plot_boxes()
        a = fig.add_subplot(224)  # projection='3d'
        plt.scatter(pys,-pxs,s=1)
        plt.show()


    return image,boxes


## drawing ####

def draw_lidar(lidar, is_grid=False, is_top_region=True, fig=None):

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))

    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)

    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if 1:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)



    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
    print(mlab.view())



def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=2):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991


# main #################################################################33
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    need_images=0

    basedir = '../data/kitti/'
    date  = '2011_09_26'
    drive = '0005'

    basedirdrive = basedir + '/' + date + '/' + date + '_drive_' + drive + '_sync/'

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive) #, range(0, 50, 5))
    print(dataset)
    # Load some data
    dataset.load_calib()         # Calibration data are accessible as named tuples
    dataset.load_timestamps()    # Timestamps are parsed into datetime objects
    dataset.load_oxts()          # OXTS packets are loaded as named tuples
    if need_images:
        #dataset.load_gray()         # Left/right images are accessible as named tuples
        dataset.load_rgb()          # Left/right images are accessible as named tuples
    dataset.load_velo()          # Each scan is a Nx4 array of [x,y,z,reflectance]

    tracklet_file = basedirdrive + 'tracklet_labels.xml'

    num_frames=len(dataset.velo)  #154
    objects = read_objects(tracklet_file, num_frames)

    ############# convert ###########################
    os.makedirs(basedirdrive+'/seg',exist_ok=True)

    if need_images:  ## rgb images --------------------
        os.makedirs(basedirdrive+'seg/rgb',exist_ok=True)

        for n in range(num_frames):
            print(n)
            rgb = dataset.rgb[n][0]
            rgb =(rgb*255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(basedirdrive+'seg/rgb/rgb_%05d.png'%n,rgb)

        exit(0)


    if 1:  ## top images --------------------
        os.makedirs(basedirdrive+'seg/lidar',exist_ok=True)
        os.makedirs(basedirdrive+'seg/top',exist_ok=True)
        os.makedirs(basedirdrive+'seg/top_image',exist_ok=True)

        for n in range(num_frames):
            print(n)
            lidar = dataset.velo[n]
            top_image = lidar_to_top(lidar,objects[n])

            np.save(basedirdrive+'seg/lidar/lidar_%05d.npy'%n,lidar)
            np.save(basedirdrive+'seg/top/top_%05d.npy'%n,top_image)
            cv2.imwrite(basedirdrive+'seg/top_image/top_image_%05d.png'%n,top_image)

        exit(0)



    if 1:  ## boxes3d  --------------------
        os.makedirs(basedirdrive+'seg/gt_boxes3d',exist_ok=True)
        os.makedirs(basedirdrive+'seg/gt_labels',exist_ok=True)
        for n in range(num_frames):
            print(n)
            objs = objects[n]
            gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objs)

            np.save(basedirdrive+'seg/gt_boxes3d/gt_boxes3d_%05d.npy'%n,gt_boxes3d)
            np.save(basedirdrive+'seg/gt_labels/gt_labels_%05d.npy'%n,gt_labels)

        exit(0)


    ############# analysis ###########################
    if 0: ## make mean
        mean_image = np.zeros((400,400),dtype=np.float32)
        num_frames=20
        for n in range(num_frames):
            print(n)
            top_image = cv2.imread(basedirdrive+'seg/top_image/top_image_%05d.png'%n,0)
            mean_image += top_image.astype(np.float32)

        mean_image = mean_image/num_frames
        cv2.imwrite(basedirdrive+'seg/top_image/top_mean_image.png',mean_image)


    if 0: ## gt_3dboxes distribution ... location and box, height
        depths =[]
        aspects=[]
        scales =[]
        mean_image = cv2.imread(basedirdrive+'seg/top_image/top_mean_image.png',0)

        for n in range(num_frames):
            print(n)
            gt_boxes3d = np.load(basedirdrive+'seg/gt_boxes3d/gt_boxes3d_%05d.npy'%n)

            top_boxes = box3d_to_top_box(gt_boxes3d)
            draw_box3d_on_top(mean_image, gt_boxes3d,color=(255,255,255), thickness=1, darken=1)


            for i in range(len(top_boxes)):
                x1,y1,x2,y2 = top_boxes[i]
                w = math.fabs(x2-x1)
                h = math.fabs(y2-y1)
                area = w*h
                s = area**0.5
                scales.append(s)

                a = w/h
                aspects.append(a)

                box3d = gt_boxes3d[i]
                d = np.sum(box3d[0:4,2])/4 -  np.sum(box3d[4:8,2])/4
                depths.append(d)

        depths  = np.array(depths)
        aspects = np.array(aspects)
        scales  = np.array(scales)

        numpy.savetxt(basedirdrive+'seg/depths.txt',depths)
        numpy.savetxt(basedirdrive+'seg/aspects.txt',aspects)
        numpy.savetxt(basedirdrive+'seg/scales.txt',scales)
        cv2.imwrite(basedirdrive+'seg/top_image/top_rois.png',mean_image)








    #----------------------------------------------------------
    #----------------------------------------------------------
    exit(0)





    #----------------------------------------------------------
    lidar = dataset.velo[0]

    objs = objects[0]
    gt_labels, gt_boxes, gt_boxes3d = obj_to_gt(objs)

    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(lidar, fig=fig)
    draw_gt_boxes3d(gt_boxes3d, fig=fig)
    mlab.show(1)

    print ('** calling lidar_to_tops() **')
    if 0:
        top, top_image = lidar_to_top(lidar)
        rgb = dataset.rgb[0][0]
    else:
        top = np.load(basedirdrive+'one_frame/top.npy')
        top_image = cv2.imread(basedirdrive+'one_frame/top_image.png')
        rgb = np.load(basedirdrive+'one_frame/rgb.npy')

    rgb =(rgb*255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # -----------


























    #check
    num = len(gt_boxes)
    for n in range(num):
       x1,y1,x2,y2 = gt_boxes[n]
       cv2.rectangle(top_image,(x1,y1), (x2,y2), (0,255,255), 1)


    ## check
    boxes3d0 = box_to_box3d(gt_boxes)

    draw_gt_boxes3d(boxes3d0,  color=(1,1,0), line_width=1, fig=fig)
    mlab.show(1)

    for n in range(num):
        qs = make_projected_box3d(gt_boxes3d[n])
        draw_projected_box3d(rgb,qs)

    imshow('rgb',rgb)
    cv2.waitKey(0)




    #save
    #np.save(basedirdrive+'one_frame/rgb.npy',rgb)
    #np.save(basedirdrive+'one_frame/lidar.npy',lidar)
    #np.save(basedirdrive+'one_frame/top.npy',top)
    #cv2.imwrite(basedirdrive+'one_frame/top_image.png',top_image)
    #cv2.imwrite(basedirdrive+'one_frame/top_image.maked.png',top_image)

    np.save(basedirdrive+'one_frame/gt_labels.npy',gt_labels)
    np.save(basedirdrive+'one_frame/gt_boxes.npy',gt_boxes)
    np.save(basedirdrive+'one_frame/gt_boxes3d.npy',gt_boxes3d)

    imshow('top_image',top_image)
    cv2.waitKey(0)

    pause











    exit(0)

    import imageio



