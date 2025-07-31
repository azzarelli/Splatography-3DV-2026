import os, sys
import torch
import sys
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
import numpy as np
import random

from gui import GUI

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", type=int, default=4000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[8000, 15999, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument('--skip-coarse', type=str, default = None)
    parser.add_argument('--view-test', action='store_true', default=False)
    parser.add_argument("--cam-config", type=str, default = "4")
    
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
        
    
    torch.autograd.set_detect_anomaly(True)
    hyp = hp.extract(args)
    initial_name = args.expname     
    name = f'{initial_name}'
    gui = GUI(
        args=args, 
        hyperparams=hyp, 
        dataset=lp.extract(args), 
        opt=op.extract(args), 
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations, 
        saving_iterations=args.save_iterations,
        ckpt_start=args.start_checkpoint, 
        debug_from=args.debug_from, 
        expname=name,
        skip_coarse=args.skip_coarse,
        view_test=args.view_test,
        use_gui=False
    )
    import torchaudio
    wave_pth = "/home/barry/Desktop/PhD/SparseViewPaper/website/assets/audio/bassist.wav"
    start_second = 25
    target_duration_sec = 10.0

    frame_size = 1024
    hop_length = 512

    # --- Load audio ---
    waveform, sample_rate = torchaudio.load(wave_pth)

    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # shape: (samples,)

    # --- Compute number of frames for exactly 10 seconds ---
    total_samples = int(sample_rate * target_duration_sec)
    num_frames = int((total_samples - frame_size) / hop_length + 1)

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Target duration: {target_duration_sec} s")
    print(f"Total samples: {total_samples}")
    print(f"Number of frames: {num_frames}")

    # --- Get segment ---
    start_sample = int(start_second * sample_rate)
    end_sample = start_sample + total_samples

    if end_sample > waveform.numel():
        raise ValueError("Audio too short to extract requested segment.")

    audio_segment = waveform[start_sample:end_sample]  # shape: (total_samples,)

    # --- Slice into overlapping frames ---
    audio_frames = audio_segment.unfold(0, frame_size, hop_length)  # shape: (num_frames, frame_size)

    
    gui.cool_video(audio_segment, sample_rate)