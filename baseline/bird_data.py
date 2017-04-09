from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from kitti_data.draw import *
from kitti_data.io import *

from net.utility.draw import *
from net.processing.boxes3d import *

import time
import imageio

# SUPPOSED RESOLUTION OF THE SCENE
TOP_X_MIN=-70
TOP_X_MAX=70   #70.4
TOP_Y_MIN=-80  #40
TOP_Y_MAX=80
TOP_Z_MIN=-7   ###
TOP_Z_MAX= 3


def lidar_to_top(lidar,objects):
    if 1: # crop the field of view
        crop=20

        TOP_X_MIN=-crop
        TOP_X_MAX=crop
        TOP_Y_MIN=-crop
        TOP_Y_MAX=crop

    # zero=time.time()
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

    N=pxs.size

    def find_highest_and_density_in_cell(pxs,pys,pzs,resolution):
        zero=time.time()
        idx_max=np.zeros((resolution,resolution),dtype=int)
        density=np.zeros((resolution,resolution))
        # preprocessed=time.time()
        # print('preprocessed in '+str(preprocessed-zero))

        cellx=np.asarray(np.floor((pxs-TOP_X_MIN)/x_scale),int)
        celly=np.asarray(np.floor((pys-TOP_Y_MIN)/y_scale),int)

        density[cellx,celly]+=1
        height_sort=pzs.argsort()
        idx_max[cellx[height_sort],celly[height_sort]]=height_sort

        # features_found=time.time()
        # print('features_found in '+str(features_found-preprocessed))
        # for i in range(0,N):
        #     cell=(math.floor((pxs[i]-TOP_X_MIN)/x_scale),math.floor((pys[i]-TOP_Y_MIN)/y_scale))
        #     if pzs[i]>pzs[idx_max[cell]]:
        #         idx_max[cell]=i

        # postprocessed = time.time()
        # print('postprocessed in ' + str(postprocessed - features_found))

        return idx_max,density

    # pixels size
    resolution=448
    x_scale=(TOP_X_MAX-TOP_X_MIN)/resolution
    y_scale=(TOP_Y_MAX-TOP_Y_MIN)/resolution

    # preprocessed=time.time()
    # print('preprocessed in '+str(preprocessed-zero))

    # find features
    idx_max, density=find_highest_and_density_in_cell(pxs,pys,pzs,resolution)

    # features_found=time.time()
    # print('features_found in '+str(features_found-preprocessed))

    has_points=density!=0
    intensity=np.zeros((resolution,resolution))
    height=np.zeros((resolution,resolution))
    intensity[has_points]=prs[idx_max][has_points]
    height[has_points]=pzs[idx_max][has_points]-TOP_Z_MIN

    # normalize density
    density=np.log(density+1)/math.log(2)
    #density[density<1]=1

    #print(np.unique(density))

    image=np.array([density, intensity*255, height*10]).transpose([1,2,0])


    # postprocessed=time.time()
    # print('postprocessed in '+str(postprocessed-features_found))

    def get_boxes(objects,x_scale, y_scale):
        boxes=np.zeros((len(objects),2,5))
        for obj in objects:
            box = obj.box[0:4, 0:2]
            box = np.r_[box, box[0:1, :]]
            box = (box - np.array([TOP_X_MIN, TOP_Y_MIN]).T) / np.array([x_scale, y_scale]).T
            box = box.T
            boxes=np.append(boxes,np.array([np.r_[box[1:2],box[0:1]]]),axis=0)
        return boxes

    boxes=get_boxes(objects,x_scale, y_scale)

    # boxes_got=time.time()
    # print('boxes_got in '+str(boxes_got))

    if 0: # plot bird views
        def plot_boxes():
            for box in boxes:
                plt.plot(box[0],box[1]) # imshow and scatter uses different axes
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


    if 0:  ## top images --------------------
        os.makedirs(basedirdrive+'seg/lidar',exist_ok=True)
        os.makedirs(basedirdrive+'seg/top',exist_ok=True)
        os.makedirs(basedirdrive+'seg/top_image',exist_ok=True)
        os.makedirs(basedirdrive+'seg/top_boxes',exist_ok=True)

        for n in range(num_frames):
            print(n)
            lidar = dataset.velo[n]
            top_image, top_boxes = lidar_to_top(lidar,objects[n])

            np.save(basedirdrive+'seg/top/top_%05d.npy'%n,top_image)
            cv2.imwrite(basedirdrive+'seg/top_image/top_image_%05d.png'%n,top_image)
            np.save(basedirdrive+'seg/top_boxes/boxes%05d.npy'%n,top_boxes)

    with imageio.get_writer('../top.gif', mode='I') as writer:
        for i in np.arange(154):
            image = imageio.imread(basedirdrive+'seg/top_image/top_image_%05d.png' % i)
            writer.append_data(image)

    with imageio.get_writer('../movie.gif', mode='I') as writer:
        for i in np.arange(154):
            image = imageio.imread(basedirdrive+'seg/rgb/rgb_%05d.png' % i)
            writer.append_data(image)