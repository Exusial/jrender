import os
from cv2 import imwrite
import jittor as jt
import numpy as np
from skimage.io import imread
from jrender.structures.utils import faces_vertices
from .load_textures import _load_textures_for_softras

def bump_mapToNormal_map(bump_image):

    xy=np.full((bump_image.shape[0]-2,bump_image.shape[1]-2),6)
    normal_image=-np.stack((bump_image[1:-1:,:-2:]-bump_image[1:-1:,2::],bump_image[:-2:,1:-1:]-bump_image[2::,1:-1:],-xy),axis=2)
    normal_image=jt.array(normal_image.copy()).float32()
    normal_image=(jt.normalize(normal_image,eps=1e-5,dim=2)+1.)/2.                
    return  normal_image

def load_mtl(filename_mtl):
    '''
    load color (Kd) and filename of textures from *.mtl
    load normal_map or bump_map from *.mtl 
    '''
    texture_filenames = {}
    normal_filenames={}
    colors = {}
    material_name = ''
    normal_type=""
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                if line.split()[0] == 'map_Kd':
                    texture_filenames[material_name] = line.split()[1]
                if line.split()[0] == 'Kd':
                    colors[material_name] = np.array(list(map(float, line.split()[1:4])))
                if line.split()[0]=='map_bump':
                    normal_type='bump_map'
                    normal_filenames[normal_type]=(line.split()[3])
                if line.split()[0]=='map_normal':
                    normal_type='normal_map'
                    normal_filenames[normal_type]=(line.split()[3])
    return colors, texture_filenames, normal_filenames


def load_textures(filename_obj, filename_mtl, texture_res,face_vertices=None):
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces for textures
    faces = []
    material_names = []
    material_name = ''
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '/' in vs[0] and '//' not in vs[0]:
                v0 = int(vs[0].split('/')[1])
            else:
                v0 = 0
            for i in range(nv - 2):
                if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[1])
                else:
                    v1 = 0
                if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[1])
                else:
                    v2 = 0
                faces.append((v0, v1, v2))
                material_names.append(material_name)
        if line.split()[0] == 'usemtl':
            material_name = line.split()[1]
    faces = np.vstack(faces).astype(np.int32) - 1
    faces = vertices[faces]                                         #face_texcoords
    faces = jt.array(faces).float32()

    colors, texture_filenames ,normal_filenames= load_mtl(filename_mtl)

    textures = jt.ones((faces.shape[0], texture_res**2, 3), dtype="float32")
    normal_textures = jt.ones((faces.shape[0], texture_res**2, 3), dtype="float32")             
    TBN=jt.ones((faces.shape[0],3,3),dtype="float32")

    for material_name, color in colors.items():
        color = jt.array(color).float32()
        for i, material_name_f in enumerate(material_names):
            if material_name == material_name_f:
                textures[i, :, :] = color.unsqueeze(0)

    #load color (Kd)  
    for material_name, filename_texture in texture_filenames.items():
        filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)
        image = imread(filename_texture).astype(np.float32) / 255.

        # texture image may have one channel (grey color)
        if len(image.shape) == 2:
            image = np.stack((image,)*3,-1)
        # or has extral alpha channel shoule ignore for now
        if image.shape[2] == 4:
            image = image[:,:,:3]

        image = image[::-1, :, :]
        image = jt.array(image.copy()).float32()
        is_update = (np.array(material_names) == material_name).astype(np.int32)
        is_update = jt.array(is_update).int()
        textures = _load_textures_for_softras(image, faces, textures, is_update)

    #load normal
    if len(normal_filenames) == 0:
        normal_textures=None
        TBN=None
    else:
        for normal_type,filename_normal in normal_filenames.items():
            filename_normal = os.path.join(os.path.dirname(filename_obj), filename_normal)
            image = imread(filename_normal).astype(np.float32) / 255.
            if(normal_type=='bump_map'):
                image=bump_mapToNormal_map(image)
                filename_bumpToNormal=os.path.join(os.path.dirname(filename_obj), 'normal_high.jpg')
                imwrite(filename_bumpToNormal,np.array(image[:,:,::-1])*255.)

            #create TBN
            e1=face_vertices[::,0]-face_vertices[::,1]
            e2=face_vertices[::,0]-face_vertices[::,2]
            n=jt.normalize(jt.cross(e1,e2).unsqueeze(1),dim=2)
            u1=(faces[::,0,0]-faces[::,1,0])
            v1=(faces[::,0,1]-faces[::,1,1])
            u2=(faces[::,0,0]-faces[::,2,0])
            v2=(faces[::,0,1]-faces[::,2,1])
            denom=jt.array(1/(u1*v2-u2*v1)).float32().unsqueeze(1).unsqueeze(2)
            inverse=jt.array(np.stack((np.stack((v2,-v1),axis=1),np.stack((-u2,u1),axis=1)),axis=1)).float32()
            e=jt.array(np.stack((e1,e2),axis=1)).float32()
            TB=jt.matmul(inverse,e)
            TB=denom*TB
            T=TB[::,0,::]
            T=T.unsqueeze(1)
            T_n=jt.sum(T*n,dim=2).unsqueeze(1)
            T=T-(T_n*n)
            T=jt.normalize(T,dim=2)
            B=jt.normalize(jt.cross(n,T),dim=2)
            TBN=jt.concat([T,B,n],dim=1)

            is_update=jt.ones(len(material_names)).int()
            image = image[::-1, :, :]
            normal_textures = _load_textures_for_softras(image,faces,normal_textures,is_update)
            normal_textures = jt.normalize((normal_textures*2-1),dim=2)
        
    return textures , normal_textures ,TBN , faces                

def load_obj(filename_obj, normalization=False, load_texture=False, texture_res=4, texture_type='surface'):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    assert texture_type in ['surface', 'vertex']

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])

    vertices = jt.array(np.vstack(vertices)).float32()
    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = jt.array(np.vstack(faces).astype(np.int32)).float32() - 1
    
    # load textures
    if load_texture and texture_type == 'surface':
        textures = None
        normal_textures=None
        for line in lines:
            if line.startswith('mtllib'):
                filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
                face_vertices=vertices[faces]
                textures ,normal_textures,TBN ,face_texcoords= load_textures(filename_obj, filename_mtl, texture_res,face_vertices)
        if textures is None:
            raise Exception('Failed to load textures.')
    elif load_texture and texture_type == 'vertex':
        textures = []
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'v':
                textures.append([float(v) for v in line.split()[4:7]])
        textures = jt.array(np.vstack(textures)).float()
    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)
        vertices /= jt.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0) / 2
    if load_texture:
        return vertices, faces, textures ,normal_textures,TBN ,face_texcoords
    else:
        return vertices, faces
