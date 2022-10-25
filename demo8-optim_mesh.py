import os
import argparse
import glob
import json
import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
import imageio

import jrender as jr
jt.flags.use_cuda = 1

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
np.random.seed(1)
pi = np.pi

def fov_to_focal_length(resolution: int, degrees: float):
    return 0.5*resolution/np.tan(0.5*degrees*pi/180)

def read_image_imageio(img_file):
    img = imageio.imread(img_file)
    img = np.asarray(img).astype(np.float32)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img / 255.0

def read_image(file):
    if os.path.splitext(file)[1] == ".bin":
        with open(file, "rb") as f:
            bytes = f.read()
            h, w = struct.unpack("ii", bytes[:8])
            img = np.frombuffer(bytes, dtype=np.float16, count=h *
                                w*4, offset=8).astype(np.float32).reshape([h, w, 4])
    else:
        img = jt.array(read_image_imageio(file))
    return img.numpy()

class Simple_NerfDataset(Dataset):
    def __init__(self, data_dir, batch_size=1, mode="test"):
        super().__init__(batch_size)
        self.data_dir = data_dir
        self.mode = mode
        self.image_data = []
        self.transforms_gpu = []
        self.focal_lengths = []
        self.metadata = []
        self.H = self.W = self.n_images = 0
        self.load_data(data_dir)
        self.total_len = self.n_images

    def __getitem__(self, idx):
        # return intrinsic, extrinsic matrix, gt
        K = jt.array([[self.focal_lengths[idx][0], 0, self.metadata[idx][4]],
                    [0, self.focal_lengths[idx][1], self.metadata[idx][5]],
                    [0,0,1]])
        return K, self.transforms_gpu[idx][:,:3], self.transforms_gpu[idx][:,3], jt.array(self.image_data[idx])

    def load_data(self,root_dir=None):
        print(f"load {self.mode} data")
        if root_dir is None:
            root_dir=self.data_dir
        ##get json file
        json_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1] == ".json":
                    if self.mode in os.path.splitext(file)[0] or (self.mode=="train" and "val" in os.path.splitext(file)[0]):
                        json_paths.append(os.path.join(root, file))
        json_data=None
        ## get frames
        for json_path in json_paths:
            with open(json_path,'r')as f:
                data=json.load(f)
            if json_data is None:
                json_data=data
            else:
                json_data['frames']+=data['frames']

        if 'h' in json_data:
            self.H=int(json_data['h'])
        if 'w' in json_data:
            self.W=int(json_data['w'])

        frames=json_data['frames']
        if self.mode=="val":
            frames=frames[::10]
        iter = 0
        for frame in tqdm(frames):
            img_path=os.path.join(self.data_dir,frame['file_path'])
            if not os.path.exists(img_path):
                img_path=img_path+".png"
                if not os.path.exists(img_path):
                    continue
            img = read_image(img_path)
            if self.H==0 or self.W==0:
                self.H=int(img.shape[0])
                self.W=int(img.shape[1])
            self.image_data.append(img)
            self.n_images+=1
            matrix=np.array(frame['transform_matrix'],np.float32)[:-1, :]
            self.transforms_gpu.append(matrix)
            iter += 1
            if iter > 3:
                break
        self.resolution=[self.W,self.H]
        # self.resolution_gpu=jt.array(self.resolution)
        metadata=np.empty([11],np.float32)
        metadata[0]=json_data.get('k1',0)
        metadata[1]=json_data.get('k2',0)
        metadata[2]= json_data.get('p1',0)
        metadata[3]=json_data.get('p2',0)
        metadata[4]=json_data.get('cx',self.W/2)/self.W
        metadata[5]=json_data.get('cy',self.H/2)/self.H
        def read_focal_length(resolution: int, axis: str):
            if 'fl_'+axis in json_data:
                return json_data['fl_'+axis]
            elif 'camera_angle_'+axis in json_data:
                return fov_to_focal_length(resolution, json_data['camera_angle_'+axis]*180/pi)
            else:
                return 0
        x_fl = read_focal_length(self.resolution[0], 'x')
        y_fl = read_focal_length(self.resolution[1], 'y')
        focal_length = []
        if x_fl != 0:
            focal_length = [x_fl, x_fl]
            if y_fl != 0:
                focal_length[1] = y_fl
        elif y_fl != 0:
            focal_length = [y_fl, y_fl]
        else:
            raise RuntimeError("Couldn't read fov.")
        self.focal_lengths.append(focal_length)
        metadata[6]=focal_length[0]
        metadata[7]=focal_length[1]

        light_dir=np.array([0,0,0])
        metadata[8:]=light_dir
        self.metadata = np.expand_dims(metadata,0).repeat(self.n_images,axis=0)
        self.H=int(self.H)
        self.W=int(self.W)

        self.image_data=jt.array(self.image_data).permute(0,3,1,2)
        self.transforms_gpu=jt.array(self.transforms_gpu)
        self.focal_lengths=jt.array(self.focal_lengths).repeat(self.n_images,1).numpy()
        ## transpose to adapt Eigen::Matrix memory
        # self.transforms_gpu=self.transforms_gpu.transpose(0,2,1)
        self.metadata=jt.array(self.metadata).numpy()
    
    def __len__(self):
        return self.n_images

class Model(nn.Module):
    def __init__(self, filename_obj, args):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = jr.Mesh.from_ply(filename_obj, dr_type='n3mr')
        self.vertices = (self.template_mesh.vertices * 0.6).stop_grad()
        self.faces = self.template_mesh.faces.stop_grad()
        self.vertex_colors = jt.array(self.template_mesh.vertex_colors).float32()
        self.vertex_colors.requires_grad = True
        # self.textures = self.template_mesh.textures
        # setup renderer
        K = jt.zeros((args.batch_size, 3, 3))
        self.renderer = jr.Renderer(image_size=800, camera_mode='projection', K=K, light_intensity_directionals=0.0, light_intensity_ambient=1.0, dr_type='n3mr')

    def execute(self, mode=None):
        image = self.renderer(self.vertices, self.faces, mode=mode, vertex_colors=self.vertex_colors)
        return image


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()

def train_one_epoch(net, train_dataset, optimizer, background_color=None):
    iter = 0
    for K, R, t, gt in tqdm(train_dataset):
        net.renderer.transform.set_projection_matrix(K, R, t)
        ret = net(mode="vertex_color")
        rgb_map = ret["rgb"]
        loss = jt.sqrt((rgb_map - gt[:,:3,...])**2).sum()
        print(rgb_map)
        optimizer.step(loss)
        # save image for each epoch
        image = rgb_map.numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % iter, image)
        iter += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'obj/spot/spot_triangulated.obj'))
    # parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'ref/ref_texture.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'results/output_optim_textures'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-dir', '--data_dir', type=str)
    parser.add_argument('-bs', '--batch_size', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.filename_output, exist_ok=True)
    train_dataset = Simple_NerfDataset(args.data_dir, args.batch_size)
    model = Model(args.filename_obj, args)

    optimizer = nn.Adam([model.vertex_colors], lr=0.1, betas=(0.5,0.999))


    # draw object
    loop = tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        train_one_epoch(model, train_dataset, optimizer)
        # model.renderer.transform.set_eyes_from_angles(2.732, 0, azimuth)
        # images = model.renderer(model.vertices, model.faces, jt.tanh(model.textures))
        # image = images.numpy()[0].transpose((1, 2, 0))
        # imsave('/tmp/_tmp_%04d.png' % num, image)
    make_gif(os.path.join(args.filename_output, 'result.gif'))


if __name__ == '__main__':
    main()