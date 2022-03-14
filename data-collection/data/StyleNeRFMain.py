file_string = ["""
import os
import re
import time
import glob
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
from renderer import Renderer

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:

    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(",")
    return [int(x) for x in vals]

#----------------------------------------------------------------------------
os.environ["PYOPENGL_PLATFORM"] = "egl"

@click.command()
@click.pass_context
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option("--seeds", type=num_range, help="List of random seeds")
@click.option("--trunc", "truncation_psi", type=float, help="Truncation psi", default=1, show_default=True)
@click.option("--class", "class_idx", type=int, help="Class label (unconditional if not specified)")
@click.option("--noise-mode", help="Noise mode", type=click.Choice(["const", "random", "none"]), default="const", show_default=True)
@click.option("--projected-w", help="Projection result file", type=str, metavar="FILE")
@click.option("--outdir", help="Where to save the output images", type=str, required=True, metavar="DIR")
@click.option("--render-program", default=None, show_default=True)
@click.option("--render-option", default=None, type=str, help="e.g. up_256, camera, depth")
@click.option("--n_steps", default=8, type=int, help="number of steps for each seed")
@click.option("--no-video", default=False)
@click.option("--relative_range_u_scale", default=1.0, type=float, help="relative scale on top of the original range u")
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    render_program=None,
    render_option=None,
    n_steps=8,
    no_video=False,
    relative_range_u_scale=1.0
):

    
    device = torch.device("cuda")
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + "/*.pkl"))[-1]
    print("Loading networks from \'%s\'..." % network_pkl)
    
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network["G_ema"].to(device) # type: ignore
        D = network["D"].to(device)
    # from fairseq import pdb;pdb.set_trace()
    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail("Must specify class label with --class when using a conditional network")
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ("warn: --class=lbl ignored when running on an unconditional network")

    # avoid persistent classes... 
    from training.networks import Generator
    # from training.stylenerf import Discriminator
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
        # D2 = Discriminator(*D.init_args, **D.init_kwargs).to(device)
        # misc.copy_params_and_buffers(D, D2, require_all=False)
    G2 = Renderer(G2, D, program=render_program)
    
    # Generate images.
    all_imgs = []

    def stack_imgs(imgs):
        img = torch.stack(imgs, dim=2)
        return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

    def proc_img(img): 
        return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    if projected_w is not None:
        ws = np.load(projected_w)
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        img = G2(styles=ws, truncation_psi=truncation_psi, noise_mode=noise_mode, render_option=render_option)
        assert isinstance(img, List)
        imgs = [proc_img(i) for i in img]
        all_imgs += [imgs]
    
    else:
        for seed_idx, seed in enumerate(seeds):
            print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
            G2.set_random_seed(seed)
            z = torch.from_numpy(np.random.RandomState(seed).randn(2, G.z_dim)).to(device)
            relative_range_u = [0.5 - 0.5 * relative_range_u_scale, 0.5 + 0.5 * relative_range_u_scale]
            outputs = G2(
                z=z,
                c=label,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                render_option=render_option,
                n_steps=n_steps,
                relative_range_u=relative_range_u,
                return_cameras=True)
            if isinstance(outputs, tuple):
                img, cameras = outputs
            else:
                img = outputs

            if isinstance(img, List):
                imgs = [proc_img(i) for i in img]
                if not no_video:
                    all_imgs += [imgs]
           
                curr_out_dir = os.path.join(outdir, "seed_{:0>6d}".format(seed))
                os.makedirs(curr_out_dir, exist_ok=True)

                if (render_option is not None) and ("gen_ibrnet_metadata" in render_option):
                    intrinsics = []
                    poses = []
                    _, H, W, _ = imgs[0].shape
                    for i, camera in enumerate(cameras):
                        intri, pose, _, _ = camera
                        focal = (H - 1) * 0.5 / intri[0, 0, 0].item()
                        intri = np.diag([focal, focal, 1.0, 1.0]).astype(np.float32)
                        intri[0, 2], intri[1, 2] = (W - 1) * 0.5, (H - 1) * 0.5

                        pose = pose.squeeze().detach().cpu().numpy() @ np.diag([1, -1, -1, 1]).astype(np.float32)
                        intrinsics.append(intri)
                        poses.append(pose)

                    intrinsics = np.stack(intrinsics, axis=0)
                    poses = np.stack(poses, axis=0)

                    np.savez(os.path.join(curr_out_dir, "cameras.npz"), intrinsics=intrinsics, poses=poses)
                        

                img_dir = os.path.join(curr_out_dir, "images_raw")
                os.makedirs(img_dir, exist_ok=True)
                for step, img in enumerate(imgs):
                    PIL.Image.fromarray(img[0].detach().cpu().numpy(), "RGB").save(f"{img_dir}/{step:03d}.png")

            else:
                img = proc_img(img)[0]
                PIL.Image.fromarray(img.numpy(), "RGB").save(f"{outdir}/seed_{seed:0>6d}.png")

    if len(all_imgs) > 0 and (not no_video):
         # write to video
        timestamp = time.strftime("%Y%m%d.%H%M%S",time.localtime(time.time()))
        seeds = ",".join([str(s) for s in seeds]) if seeds is not None else "projected"
        network_pkl = network_pkl.split("/")[-1].split(".")[0]
        all_imgs = [stack_imgs([a[k] for a in all_imgs]).numpy() for k in range(len(all_imgs[0]))]
        imageio.mimwrite(f"{outdir}/{network_pkl}_{timestamp}_{seeds}.mp4", all_imgs, fps=30, quality=8)
        outdir = f"{outdir}/{network_pkl}_{timestamp}_{seeds}"
        os.makedirs(outdir, exist_ok=True)
        for step, img in enumerate(all_imgs):
            PIL.Image.fromarray(img, "RGB").save(f"{outdir}/{step:04d}.png")


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

""",

'''
import gradio as gr
import numpy as np
import dnnlib
import time
import legacy
import torch
import glob
import os
import cv2
import tempfile
import imageio

from torch_utils import misc
from renderer import Renderer
from training.networks import Generator

device = torch.device('cuda')
render_option = 'freeze_bg,steps36'


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_camera_traj(model, pitch, yaw, fov=12, batch_size=1, model_name='FFHQ512'):
    gen = model.synthesis
    range_u, range_v = gen.C.range_u, gen.C.range_v
    if not (('car' in model_name) or ('Car' in model_name)):  # TODO: hack, better option?
        yaw, pitch = 0.5 * yaw, 0.3  * pitch
        pitch = pitch + np.pi/2
        u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
        v = (pitch - range_v[0]) / (range_v[1] - range_v[0])
    else:
        u = (yaw + 1) / 2
        v = (pitch + 1) / 2
    cam = gen.get_camera(batch_size=batch_size, mode=[u, v, 0.5], device=device, fov=fov)
    return cam


def check_name(model_name='FFHQ512'):
    """Gets model by name."""
    if model_name == 'FFHQ512':
        network_pkl = "./pretrained/ffhq_512.pkl"
    elif model_name == 'FFHQ512v2':
        network_pkl = "./pretrained/ffhq_512.v2.pkl"
    elif model_name == 'AFHQ512':
        network_pkl = "./pretrained/afhq_512.pkl"
    elif model_name == 'MetFaces512':
        network_pkl = "./pretrained/metfaces_512.pkl"
    elif model_name == 'CompCars256':
        network_pkl = "./pretrained/cars_256.pkl"
    elif model_name == 'FFHQ1024':
        network_pkl = "./pretrained/ffhq_1024.pkl"
    else:
        if os.path.isdir(model_name):
            network_pkl = sorted(glob.glob(model_name + '/*.pkl'))[-1]
        else:
            network_pkl = model_name
    return network_pkl


def get_model(network_pkl):
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device)  # type: ignore

    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)

    print('compile and go through the initial image')
    G2 = G2.eval()
    
    init_z = torch.from_numpy(np.random.RandomState(0).rand(1, G2.z_dim)).to(device)
    init_cam = get_camera_traj(G2, 0, 0, model_name=network_pkl)
    dummy = G2(z=init_z, c=None, camera_matrices=init_cam, render_option=render_option, theta=0)
    res = dummy['img'].shape[-1]
    imgs = [None, None]
    return G2, res, imgs


global_states = list(get_model(check_name()))
wss  = [None, None]

def proc_seed(history, seed):
    if isinstance(seed, str):
        seed = 0
    else:
        seed = int(seed)

def stack_imgs(imgs):
    img = torch.stack(imgs, dim=2)
    return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

def proc_img(img): 
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

def f_synthesis(model_name, program, model_find, trunc, seed1, seed2, mix1, mix2, roll, fov):
    history = gr.get_state() or {}
    seeds = []
    
    if model_find != "":
        model_name = model_find

    model_name = check_name(model_name)
    if model_name != history.get("model_name", None):
        model, res, imgs = get_model(model_name)
        global_states[0] = model
        global_states[1] = res
        global_states[2] = imgs

    model, res, imgs = global_states
    if program  == 'image':
        program = 'rotation_camera3'
    elif program == 'image+normal':
        program = 'rotation_both'
    renderer = Renderer(model, None, program=program)

    for idx, seed in enumerate([seed1, seed2]):
        if isinstance(seed, str):
            seed = 0
        else:
            seed = int(seed)
        
        if (seed != history.get(f'seed{idx}', -1)) or \
            (model_name != history.get("model_name", None)) or \
            (trunc != history.get("trunc", 0.7)) or \
            (wss[idx] is None):
            print(f'use seed {seed}')
            set_random_seed(seed)
            with torch.no_grad():
                z   = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, model.z_dim).astype('float32')).to(device)
                ws  = model.mapping(z=z, c=None, truncation_psi=trunc)
                imgs[idx] = [proc_img(i) for i in renderer(styles=ws, render_option=render_option)]
                ws  = ws.detach().cpu().numpy()
            wss[idx]  = ws
        else:
            seed = history[f'seed{idx}']
        
        seeds += [seed]
        history[f'seed{idx}'] = seed

    history['trunc'] = trunc
    history['model_name'] = model_name
    gr.set_state(history)
    set_random_seed(sum(seeds))

    # style mixing (?)
    ws1, ws2 = [torch.from_numpy(ws).to(device) for ws in wss]
    ws = ws1.clone()
    ws[:, :8] = ws1[:, :8] * mix1 + ws2[:, :8] * (1 - mix1)
    ws[:, 8:] = ws1[:, 8:] * mix2 + ws2[:, 8:] * (1 - mix2)

    
    dirpath = tempfile.mkdtemp()
    start_t = time.time()
    with torch.no_grad():
        outputs  = [proc_img(i) for i in renderer(
            styles=ws.detach(), 
            theta=roll * np.pi,
            render_option=render_option)]
        all_imgs = [imgs[0], outputs, imgs[1]]
        all_imgs = [stack_imgs([a[k] for a in all_imgs]).numpy() for k in range(len(all_imgs[0]))]
        imageio.mimwrite(f'{dirpath}/output.mp4', all_imgs, fps=30, quality=8)
    end_t = time.time()
    print(f'rendering time = {end_t-start_t:.4f}s')
    return f'{dirpath}/output.mp4'

model_name = gr.inputs.Dropdown(['FFHQ512', 'FFHQ512v2', 'AFHQ512', 'MetFaces512', 'CompCars256', 'FFHQ1024'])
model_find = gr.inputs.Textbox(label="checkpoint path", default="")
program = gr.inputs.Dropdown(['image', 'image+normal'], default='image')
trunc  = gr.inputs.Slider(default=0.7, maximum=1.0, minimum=0.0, label='truncation trick')
seed1  = gr.inputs.Number(default=1, label="seed1")
seed2  = gr.inputs.Number(default=9, label="seed2")
mix1   = gr.inputs.Slider(minimum=0, maximum=1, default=0, label="linear mixing ratio (geometry)")
mix2   = gr.inputs.Slider(minimum=0, maximum=1, default=0, label="linear mixing ratio (apparence)")
roll   = gr.inputs.Slider(minimum=-1, maximum=1, default=0, label="roll (optional, not suggested)")
fov    = gr.inputs.Slider(minimum=9, maximum=15, default=12, label="fov")
css = ".output_video {height: 40rem !important; width: 100% !important;}"
gr.Interface(fn=f_synthesis,
             inputs=[model_name, program, model_find, trunc, seed1, seed2, mix1, mix2, roll, fov],
             outputs="video",
             layout='unaligned',
             server_port=20011,
             css=css).launch()

''',

'''

from genericpath import exists
import numpy as np 
import os
import glob
import torch
import json
import click
import dnnlib
import legacy
import copy
import PIL.Image
import glob
import tqdm
import time
import imageio
import cv2
import torch.nn.functional as F
from torchvision.utils import save_image
from training.networks import Generator
from renderer import Renderer
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

device  = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DECODERPATH = "/checkpoint/jgu/space/gan/ffhq/debug3/00486-nores_critical_bgstyle-ffhq_512-mirror-paper512-stylenerf_pgc-noaug"
ENCODERPATH = DECODERPATH + '/encoder3.1/checkpoints'

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default=DECODERPATH)
@click.option('--encoder', 'encoder_pkl', help='pre-trained encoder for initialization', default=ENCODERPATH)
@click.option('--target',  'target_path', help='Target image file to project to', required=True, metavar='DIR')
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--no_image', default=False, type=bool)
def main(
    network_pkl: str,
    encoder_pkl: str,
    target_path: str,
    outdir: str,
    seed: int,
    no_image
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    conv2d_gradfix.enabled = True  # Improves training speed.

    # Load networks.
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    E = None
    if encoder_pkl is not None:
        if os.path.isdir(encoder_pkl):
            encoder_pkl = sorted(glob.glob(encoder_pkl + '/*.pkl'))[-1]
        print('Loading pretrained encoder from "%s"...' % encoder_pkl)
        with dnnlib.util.open_url(encoder_pkl) as fp:
            E = legacy.load_network_pkl(fp)['E'].requires_grad_(False).to(device) # type: ignore
    try:
        with torch.no_grad():
            G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
            misc.copy_params_and_buffers(G, G2, require_all=False)
    except RuntimeError:
        G2 = G
    G2 = Renderer(G2, None, program=None)

    # Output files
    inferred_poses = {}
    target_files   = sorted(glob.glob(target_path + '/*.png'))
    timestamp      = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))

    if not no_image:
        video      = imageio.get_writer(f'{outdir}/proj_{timestamp}.mp4', mode='I', fps=4, codec='libx264', bitrate='16M')

    for step, target_fname in enumerate(tqdm.tqdm(target_files)):
        target_id0 = target_fname.split('/')[-1]
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target_image = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

        ws, cm = E(target_image[None,:].to(torch.float32) / 127.5 - 1)
        target_image = target_image.clone().unsqueeze(0).to(torch.float32) / 255.
        opt_weights  = [{'params': ws}]
        kwargs = G2.get_additional_params(ws)
        kwargs['camera_matrices']  = G.synthesis.get_camera(1, device, mode=cm)
        inferred_poses[target_id0] = kwargs['camera_matrices'][1].cpu().numpy().reshape(-1).tolist()
        # if step > 200: break
        if not no_image:
            if len(kwargs) > 0:
                # latent codes for the background
                if len(kwargs['latent_codes'][2].size()) > 0:
                    kwargs['latent_codes'][2].requires_grad = True
                    opt_weights += [{'params': kwargs['latent_codes'][2]}]
                if len(kwargs['latent_codes'][3].size()) > 0:
                    kwargs['latent_codes'][3].requires_grad = True
                    opt_weights += [{'params': kwargs['latent_codes'][3]}]

            synth_image = G2(styles=ws, **kwargs)
            synth_image = (synth_image + 1.0) / 2.0
            image = torch.cat([target_image, synth_image], -1).clamp(0,1)[0]
            image = (image.permute(1,2,0).detach().cpu().numpy() * 255).astype('uint8')
            image = cv2.resize(image, (256, 128), interpolation=cv2.INTER_AREA)
            video.append_data(image)
            # save_image(torch.cat([target_image, synth_image], -1).clamp(0,1), f"{outdir}/{target_id0}.png")

    json.dump(inferred_poses, open(f'{outdir}/extracted_poses.json', 'w'))
    print('done')
    if not no_image: video.close()

if __name__ == "__main__":
    main()

''',

'''
import os
import glob
import re
from typing import List

import click
from numpy.lib.function_base import interp
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    print(s)
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def lerp(t, v0, v1):
    v2 = (1.0 - t) * v0 + t * v1
    return v2


# Taken and adapted from wikipedia's slerp article
# https://en.wikipedia.org/wiki/Slerp
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2


# Helper function for interpolation
def interpolate(v0, v1, n_steps, interp_type='spherical', smooth=False):
    # Get the timesteps
    t_array = np.linspace(0, 1, num=n_steps, endpoint=False).reshape(-1, 1)
    if smooth:
        # Smooth interpolation, constructed following
        # https://math.stackexchange.com/a/1142755
        t_array = t_array**2 * (3 - 2 * t_array)
    
    # TODO: no need of a for loop; this can be optimized using the fact that they're numpy arrays!
    vectors = list()
    for t in t_array:
        if interp_type == 'linear':
            v = lerp(t, v0, v1)
        elif interp_type == 'spherical':
            v = slerp(t, v0, v1)
        vectors.append(v)
    
    return np.asarray(vectors)

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', 'seeds', type=num_range, help='Random seeds to use for interpolation', required=True)
@click.option('--steps', 'n_steps', type=int, default=120)
@click.option('--interp_type', 'interp_type', help='linear or spherical', default='spherical', show_default=True)
@click.option('--interp_space', 'interp_space', default='z', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
@click.option('--render-program', default=None, show_default=True)
@click.option('--render-option', default=None, type=str, help="e.g. up256, camera, depth")
def generate_interpolation(
    network_pkl: str,
    seeds: List[int],
    interp_type: str,
    interp_space: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    n_steps: int,
    render_program=None,
    render_option=None,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # avoid persistent classes... 
    from training.networks import Generator
    from renderer import Renderer
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G2 = Renderer(G2, None, program=None)
    w_avg = G2.generator.mapping.w_avg

    print('Generating W vectors...')
    all_z  = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in seeds])
    all_w  = G2.generator.mapping(torch.from_numpy(all_z).to(device), None)

    # copy the same w
    # k, m = 20, 20
    # for i in range(len(all_w)):
    #     all_w[i, :m] = all_w[0, :m]
    #     # all_w[i, k:] = all_w[0, k:]
    # from fairseq import pdb;pdb.set_trace()

    kwargs = G2.get_additional_params(all_w)

    if interp_space == 'z':
        print('Interpolation in Z space')
        interp_z = [interpolate(all_z[t], all_z[t+1], n_steps=n_steps, interp_type=interp_type) for t in range(len(seeds) - 1)]
        interp_z = np.concatenate(interp_z, 0)
    elif interp_space == 'w':
        print('Interpolation in W space')
        all_w = all_w.cpu().numpy()
        interp_w = [interpolate(all_w[t], all_w[t+1], n_steps=n_steps, interp_type=interp_type) for t in range(len(seeds) - 1)]
        interp_w = np.concatenate(interp_w, 0)
    else:
        raise NotImplementedError

    interp_codes = None
    if kwargs.get('latent_codes', None) is not None:
        codes = kwargs['latent_codes']
        interp_codes = []
        for c in codes:
            if len(c.size()) != 0:
                c = c.cpu().numpy()
                c = [interpolate(c[t], c[t+1], n_steps=n_steps, interp_type=interp_type) for t in range(len(seeds) - 1)]
                interp_codes += [torch.from_numpy(np.concatenate(c, 0)).float().to(device)]
            else:
                interp_codes += [c]
    
    batch_size = 20    
    interp_images = []
    if render_program == 'rotation':
        tspace = np.linspace(0, 1, 120)
    else:
        tspace = np.zeros(10)

    for s in range(0, (len(seeds)-1) * n_steps, batch_size):
        if interp_space == 'z':
            all_z = interp_z[s: s + batch_size]
            all_w = G2.generator.mapping(torch.from_numpy(all_z).to(device), None)
        elif interp_space == 'w':
            all_w = interp_w[s: s + batch_size]
            all_w = torch.from_numpy(all_w).to(device)
            
        all_w = w_avg + (all_w - w_avg) * truncation_psi
        if interp_codes is not None:
            kwargs = {}
            # kwargs['latent_codes'] = tuple([c[s: s + batch_size] if (len(c.size())>0) else c for c in interp_codes])
            kwargs['latent_codes'] = tuple([c[:1].repeat(all_w.size(0), 1) if (len(c.size())>0) else c for c in interp_codes])
            cams = [G2.get_camera_traj(tspace[st % tspace.shape[0]], device=all_w.device) for st in range(s, s+all_w.size(0))]

            kwargs['camera_matrices'] = tuple([torch.cat([cams[j][i] for j in range(len(cams))], 0)
            if isinstance(cams[0][i], torch.Tensor) else cams[0][i]
            for i in range(len(cams[0]))])
       
        print(f'Generating images...{s}')
        all_images = G2(styles=all_w, noise_mode=noise_mode, render_option=render_option, **kwargs)

        def proc_img(img): 
            return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

        if isinstance(all_images, List):
            all_images = torch.stack([proc_img(i) for i in all_images], dim=-1).numpy()
        else:
            all_images = proc_img(all_images).numpy()
        
        interp_images += [img for img in all_images]

    print('Saving image/video grid...')
    import imageio, time
    timestamp = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))
    network_pkl = network_pkl.split('/')[-1].split('.')[0]
    imageio.mimwrite(f'{outdir}/interp_{network_pkl}_{timestamp}.mp4', interp_images, fps=30, quality=8)
    
    outdir = f'{outdir}/{network_pkl}_{timestamp}_{seeds}'
    os.makedirs(outdir, exist_ok=True)
    for step, img in enumerate(interp_images):
        PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{step:04d}.png')
    # from fairseq import pdb;pdb.set_trace()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    with torch.no_grad():
        generate_interpolation() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
''',

'''


import numpy as np 
import matplotlib.pyplot as plt 
import os
import glob
import imageio
import torch
import torch.nn as nn

from torchvision import models 
import torch.nn.functional as F

import torch.optim as optim
import click
import dnnlib
import legacy
import copy
import pickle
import PIL.Image

from collections import OrderedDict
from torchvision.utils import save_image
from training.networks import Generator, ResNetEncoder
from renderer import Renderer
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--encoder', 'encoder_pkl', help='pre-trained encoder for initialization', default=None)
@click.option('--encoder_z', 'ez',        help='use encoder to predict z', type=bool, default=False)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--l2_lambda', default=1, type=float)
@click.option('--pl_lambda', default=1, type=float)
def main(
    network_pkl: str,
    encoder_pkl: str,
    ez: bool,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    l2_lambda: float,
    pl_lambda: float,
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    conv2d_gradfix.enabled = True  # Improves training speed.

    # Load networks.
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    
    E = None
    if encoder_pkl is not None:
        if os.path.isdir(encoder_pkl):
            encoder_pkl = sorted(glob.glob(encoder_pkl + '/*.pkl'))[-1]
        print('Loading pretrained encoder from "%s"...' % encoder_pkl)
        with dnnlib.util.open_url(encoder_pkl) as fp:
            E = legacy.load_network_pkl(fp)['E'].requires_grad_(False).to(device) # type: ignore

    try:
        with torch.no_grad():
            G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
            misc.copy_params_and_buffers(G, G2, require_all=False)
    except RuntimeError:
        G2 = G
    
    G2 = Renderer(G2, None, program=None)

    # Load target image.
    if 'gen' != target_fname[:3]:
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target_image = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)
    else:
        z = np.random.RandomState(int(target_fname[3:])).randn(1, G.z_dim)
        t = np.random.rand() if E is not None else 0
        camera_matrices = G2.get_camera_traj(t, 1, device=device)
        target_image = G2(torch.from_numpy(z).to(device), None, camera_matrices=camera_matrices)[0]
        target_image = ((target_image * 0.5 + 0.5) * 255).clamp(0,255).to(torch.uint8)

    if E is None:  # starting from initial
        z_samples = np.random.RandomState(123).randn(10000, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        w_samples = w_samples.mean(0, keepdim=True)
        ws = w_samples.clone()
        ws.requires_grad = True
        cm = None
    else:
        if not ez:
            ws, cm = E(target_image[None,:].to(torch.float32) / 127.5 - 1)
        else:
            # from fairseq import pdb;pdb.set_trace()
            zs, cm = E(target_image[None,:].to(torch.float32) / 127.5 - 1)
            ws = G.mapping(zs, None)

        ws = ws.clone()
        ws.requires_grad = True

    MSE_Loss        = nn.MSELoss(reduction="mean")
    # MSE_Loss        = nn.SmoothL1Loss(reduction='mean')
    perceptual_net  = VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device)
    target_image    = target_image.clone().unsqueeze(0).to(torch.float32) / 255.
    target_image_p  = F.interpolate(target_image, size=(256, 256), mode='area')
    target_features = perceptual_net(target_image_p)

    opt_weights = [{'params': ws}]
    kwargs = G2.get_additional_params(ws)
    if cm is not None:
        kwargs['camera_matrices'] = G.synthesis.get_camera(1, device, mode=cm)

    if len(kwargs) > 0:
        # latent codes for the background
        if len(kwargs['latent_codes'][2].size()) > 0:
            kwargs['latent_codes'][2].requires_grad = True
            opt_weights += [{'params': kwargs['latent_codes'][2]}]
        if len(kwargs['latent_codes'][3].size()) > 0:
            kwargs['latent_codes'][3].requires_grad = True
            opt_weights += [{'params': kwargs['latent_codes'][3]}]

    optimizer = optim.Adam(opt_weights, lr=0.01, betas=(0.9,0.999), eps=1e-8)
    
    print("Start...")
    loss_list = []
    
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        import time
        timestamp = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))
        video = imageio.get_writer(f'{outdir}/proj_{timestamp}.mp4', mode='I', fps=24, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')

    for i in range(num_steps):
        optimizer.zero_grad()
        # kwargs['camera_matrices'] = G.synthesis.get_camera(1, device, cs)
        synth_image = G2(styles=ws, **kwargs)
        synth_image = (synth_image + 1.0) / 2.0
        
        mse_loss, perceptual_loss = caluclate_loss(
            synth_image, target_image, target_features, perceptual_net, MSE_Loss)
        mse_loss = mse_loss * l2_lambda
        perceptual_loss = perceptual_loss * pl_lambda
        loss= mse_loss + perceptual_loss
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        loss_p  = perceptual_loss.detach().cpu().numpy()
        loss_m  = mse_loss.detach().cpu().numpy()
        loss_list.append(loss_np)

        if i % 10 == 0:
            print("iter {}: loss -- {:.5f} \t mse_loss --{:.5f} \t percep_loss --{:.5f}".format(i,loss_np,loss_m,loss_p))
            if save_video:
                image = torch.cat([target_image, synth_image], -1)
                image = image.permute(0, 2, 3, 1) * 255.
                image = image.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(image)
            
        if i % 100 == 0:
            save_image(torch.cat([target_image, synth_image], -1).clamp(0,1), f"{outdir}/{i}.png")
            np.save("loss_list.npy",loss_list)
            np.save(f"{outdir}/latent_W_{i}.npy", ws.detach().cpu().numpy())
    
    np.save(f"{outdir}/latent_last.npy", ws.detach().cpu().numpy())
    # render the learned model
    if len(kwargs) > 0:  # stylenerf
        assert save_video
        G2.program = 'rotation_camera3'
        all_images = G2(styles=ws)
        def proc_img(img): 
            return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

        target_image = proc_img(target_image * 2 - 1).numpy()[0]
        all_images = torch.stack([proc_img(i) for i in all_images], dim=-1).numpy()[0]
        for i in range(all_images.shape[-1]):
            video.append_data(np.concatenate([target_image, all_images[..., i]], 1))
        
        outdir = f'{outdir}/proj_{timestamp}'
        os.makedirs(outdir, exist_ok=True)
        for step in range(all_images.shape[-1]):
            img = all_images[..., i]
            PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{step:04d}.png')



    if save_video:
        video.close()
    
   
        
def caluclate_loss(synth_image, target_image, target_features, perceptual_net, MSE_Loss):
     #calculate MSE Loss
     mse_loss = MSE_Loss(synth_image, target_image) # (lamda_mse/N)*||G(w)-I||^2

     #calculate Perceptual Loss
     real_0, real_1, real_2, real_3 = target_features
     synth_image_p = F.interpolate(synth_image, size=(256, 256), mode='area')
     synth_0, synth_1, synth_2, synth_3 = perceptual_net(synth_image_p)
     perceptual_loss = 0
     perceptual_loss += MSE_Loss(synth_0, real_0)
     perceptual_loss += MSE_Loss(synth_1, real_1)
     perceptual_loss += MSE_Loss(synth_2, real_2)
     perceptual_loss += MSE_Loss(synth_3, real_3)
     return mse_loss, perceptual_loss

class VGG16_for_Perceptual(torch.nn.Module):
    def __init__(self,requires_grad=False,n_layers=[2,4,14,21]):
        super(VGG16_for_Perceptual,self).__init__()
        vgg_pretrained_features=models.vgg16(pretrained=True).features 

        self.slice0=torch.nn.Sequential()
        self.slice1=torch.nn.Sequential()
        self.slice2=torch.nn.Sequential()
        self.slice3=torch.nn.Sequential()

        for x in range(n_layers[0]):#relu1_1
            self.slice0.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[0],n_layers[1]): #relu1_2
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[1],n_layers[2]): #relu3_2
            self.slice2.add_module(str(x),vgg_pretrained_features[x])

        for x in range(n_layers[2],n_layers[3]):#relu4_2
            self.slice3.add_module(str(x),vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False
        
    def forward(self,x):
        h0=self.slice0(x)
        h1=self.slice1(h0)
        h2=self.slice2(h1)
        h3=self.slice3(h2)

        return h0,h1,h2,h3


if __name__ == "__main__":
    main()

''',

'''

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    from training.networks import Generator
    from renderer import Renderer
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G2 = Renderer(G2, None, program=None)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    
    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])

        if step == 0:
            kwargs = G2.get_additional_params(ws)

        synth_images = G2(styles=ws, noise_mode='const', **kwargs)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    if os.path.isdir(network_pkl):
        import glob
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
''',

'''

import os
import glob
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    print(s)
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--rows', 'row_seeds', type=num_range, help='Random seeds to use for image rows', required=True)
@click.option('--cols', 'col_seeds', type=num_range, help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=num_range, help='Style layer range', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
@click.option('--render-program', default=None, show_default=True)
@click.option('--render-option', default=None, type=str, help="e.g. up256, camera, depth")
def generate_style_mix(
    network_pkl: str,
    row_seeds: List[int],
    col_seeds: List[int],
    col_styles: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    render_program=None,
    render_option=None,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # avoid persistent classes... 
    from training.networks import Generator
    from renderer import Renderer
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G2 = Renderer(G2, None, program=render_program)

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = G2.generator.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G2.generator.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}
    # from fairseq import pdb;pdb.set_trace()
    print('Generating images...')
    all_images = G2(styles=all_w, noise_mode=noise_mode)

    def proc_img(img): 
        return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    if isinstance(all_images, List):
        all_images = torch.stack([proc_img(i) for i in all_images], dim=-1).numpy()
    else:
        all_images = proc_img(all_images).numpy()
    
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}
    
    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = G2(styles=w[np.newaxis], noise_mode=noise_mode, render_option=render_option)
            if isinstance(image, List):
                image = torch.stack([proc_img(i) for i in image], dim=-1).numpy()
            else:
                image = proc_img(image).numpy()
            image_dict[(row_seed, col_seed)] = image[0]

    # from fairseq import pdb;pdb.set_trace()
    # print('Saving images...')
    # os.makedirs(outdir, exist_ok=True)
    # for (row_seed, col_seed), image in image_dict.items():
    #     PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')

    print('Saving image/video grid...')
    import imageio, time
    timestamp = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))
    W = G.img_resolution
    H = G.img_resolution
    if image_dict[(row_seeds[0],col_seeds[0])].ndim == 4:
        STEP = image_dict[(row_seeds[0],col_seeds[0])].shape[-1]
    else:
        STEP = 1
    
    all_images = []
    for step in range(STEP):
        canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
        for row_idx, row_seed in enumerate([0] + row_seeds):
            for col_idx, col_seed in enumerate([0] + col_seeds):
                if row_idx == 0 and col_idx == 0:
                    continue
                key = (row_seed, col_seed)
                if row_idx == 0:
                    key = (col_seed, col_seed)
                if col_idx == 0:
                    key = (row_seed, row_seed)
                img = image_dict[key]
                img = img if img.ndim == 3 else img[:,:,:,step]
                canvas.paste(PIL.Image.fromarray(img, 'RGB'), (W * col_idx, H * row_idx))
        all_images.append(canvas)
    
    if STEP == 1:
        canvas.save(f'{outdir}/grid_{timestamp}.png')
    else:
        network_pkl = network_pkl.split('/')[-1].split('.')[0]
        imageio.mimwrite(f'{outdir}/grid_{network_pkl}_{timestamp}.mp4', [np.asarray(img) for img in all_images], fps=30, quality=8)
        outdir = f'{outdir}/{network_pkl}_{timestamp}'
        os.makedirs(outdir, exist_ok=True)
        for step, img in enumerate(all_images):
            img = np.asarray(img)
            PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{step:04d}.png')
#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_style_mix() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
''',

'''

import numpy as np 
import matplotlib.pyplot as plt 
import os
import glob
import imageio
import torch
import torch.nn as nn
import clip
import math

from torchvision import models 
import torch.nn.functional as F

import torch.optim as optim
import click
import dnnlib
import legacy
import copy
import PIL.Image

from collections import OrderedDict
from tqdm import tqdm
from torchvision.utils import save_image
from training.networks import Generator
from renderer import Renderer
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@click.command()
@click.option('--description', 'text',    help='the text that guides the generation', default="a person with purple hair")
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--lr_rampup', type=float, default=0.05)
@click.option('--lr_init',   type=float, default=0.1)
@click.option('--l2_lambda', type=float, default=0.008)
@click.option('--id_lambda', type=float, default=0.000)
@click.option('--trunc', type=float, default=0.7)
@click.option('--mode', type=click.Choice(['free', 'edit']), default='edit')
def main(
    text: str,
    network_pkl: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    lr_init: float,
    lr_rampup: float,
    l2_lambda: float,
    id_lambda: float,
    trunc: 0.7,
    mode: str,
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    conv2d_gradfix.enabled = True  # Improves training speed.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/edit.mp4', mode='I', fps=24, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/edit.mp4"')

    # Load networks.
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G2 = Renderer(G2, None, program=None)

    # start from an average 
    z = np.random.RandomState(seed).randn(1, G.z_dim)
    camera_matrices = G2.get_camera_traj(0, 1, device=device)
    ws_init = G.mapping(torch.from_numpy(z).to(device), None, truncation_psi=trunc)    
    initial_image = G2(styles=ws_init, camera_matrices=camera_matrices)

    ws = ws_init.clone()
    ws.requires_grad = True
    clip_loss   = CLIPLoss(stylegan_size=G.img_resolution)
    if id_lambda > 0:
        id_loss = IDLoss()
    optimizer   = optim.Adam([ws], lr=lr_init, betas=(0.9,0.999), eps=1e-8)
    pbar        = tqdm(range(num_steps))
    text_input  = torch.cat([clip.tokenize(text)]).to(device)

    for i in pbar:
        # t = i / float(num_steps)
        # lr = get_lr(t, lr_init, rampup=lr_rampup)
        # optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad()

        img_gen = G2(styles=ws, camera_matrices=camera_matrices)
        c_loss = clip_loss(img_gen, text_input)
    
        if id_lambda > 0:
            i_loss = id_loss(img_gen, initial_image)[0]
        else:
            i_loss = 0

        if mode == "edit":
            l2_loss = ((ws - ws_init) ** 2).sum()
            loss = c_loss + l2_lambda * l2_loss + id_lambda * i_loss
        else:
            l2_loss = 0
            loss = c_loss

        loss.backward()
        optimizer.step()
        pbar.set_description((f"loss: {loss.item():.4f}; c:{c_loss.item():.4f}; l2:{l2_loss:.4f}; id:{i_loss:.4f}"))
        if i % 10 == 0:
            if save_video:
                image = torch.cat([initial_image, img_gen], -1) * 0.5 + 0.5
                image = image.permute(0, 2, 3, 1) * 255.
                image = image.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(image)
            
        if i % 100 == 0:
            save_image(torch.cat([initial_image, img_gen], -1).clamp(-1,1), f"{outdir}/{i}.png", normalize=True, range=(-1, 1))
            # np.save("latent_W/{}.npy".format(name),dlatent.detach().cpu().numpy())
        
    # # render the learned model
    # if len(kwargs) > 0:  # stylenerf
    #     assert save_video
    #     G2.program = 'rotation_camera3'
    #     all_images = G2(styles=ws)
    #     def proc_img(img): 
    #         return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    #     initial_image = proc_img(initial_image * 2 - 1).numpy()[0]
    #     all_images = torch.stack([proc_img(i) for i in all_images], dim=-1).numpy()[0]
    #     for i in range(all_images.shape[-1]):
    #         video.append_data(np.concatenate([initial_image, all_images[..., i]], 1))
        
    if save_video:
        video.close()
        

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


class CLIPLoss(torch.nn.Module):
    def __init__(self, stylegan_size):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image, text):
        
        def preprocess_tensor(x):
            import torchvision.transforms.functional as F
            x = F.resize(x, size=224, interpolation=PIL.Image.BICUBIC)
            x = F.normalize(x, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            return x

        image = preprocess_tensor(image)
        # image = self.avg_pool(self.upsample(image))
        # image = self.preprocess(image)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        from training.facial_recognition.model_irse import Backbone

        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load('/private/home/jgu/.torch/models/model_ir_se50.pth'))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count

if __name__ == "__main__":
    main()
''',

'''

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
import glob
#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels, _indices in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, jit=False, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # HACK:
    # other_data = "/checkpoint/jgu/space/gan/ffhq/giraffe_results/gen_images"
    # other_data = "/checkpoint/jgu/space/gan/cars/gen_images_380000"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/FFHQEvalOutput2"
    # other_data = "/private/home/jgu/work/pi-GAN/Baselines/AFHQEvalOutput"
    # other_data = sorted(glob.glob(f'{other_data}/*.jpg'))
    # other_data = '/private/home/jgu/work/giraffe/out/afhq256/fid_images.npy'
    # other_images = np.load(other_data)
    #　from fairseq import pdb;pdb.set_trace()
    # print(f'other data size = {len(other_data)}')
    other_data = None

    # Image generation func.
    def run_generator(z, c):
        # from fairseq import pdb;pdb.set_trace()
        if hasattr(G, 'get_final_output'):
            img = G.get_final_output(z=z, c=c, **opts.G_kwargs)
        else:
            img = G(z=z, c=c, **opts.G_kwargs)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, G.z_dim], device=opts.device)
        c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c], check_trace=False)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    till_now = 0
    while not stats.is_full():
        images = []
        if other_data is None:
            for _i in range(batch_size // batch_gen):
                z = torch.randn([batch_gen, G.z_dim], device=opts.device)
                c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_gen)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                img = run_generator(z, c)
                images.append(img)
            images = torch.cat(images)
        else:
            batch_idxs = [((till_now + i) * opts.num_gpus + opts.rank) % len(other_images) for i in range(batch_size)]
            import imageio
            till_now += batch_size
            images = other_images[batch_idxs]
            images = torch.from_numpy(images).to(opts.device)
            # images = np.stack([imageio.imread(other_data[i % len(other_data)]) for i in batch_idxs], axis=0)
            # images = torch.from_numpy(images).to(opts.device).permute(0,3,1,2)
            
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------
'''
]