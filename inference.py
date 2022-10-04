import logging
import argparse

import torch
import numpy as np
from thop import profile
from pgmpy.inference import Mplp

from models.mrf import MRF 
from models.mrf_unet import MRFSuperNet, ChildNet


def get_args():
    parser = argparse.ArgumentParser(description='Inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target', type=float, default=2.4)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--lam', type=float, default=10)
    parser.add_argument('--gamma-min', type=float, default=0)
    parser.add_argument('--gamma-max', type=float, default=1e-5)
    parser.add_argument('--gamma-iter', type=int, default=20)
    parser.add_argument('--flops-path', type=str, default="flops.pkl")
    parser.add_argument('--ckp-path', type=str, default="../outputs/search/checkpoints/checkpoint050.pth")
    
    args = parser.parse_args()
    
    args.image_channels = 3
    args.num_classes = 6
    args.channel_step = 5
    args.lams = [args.lam] * (args.m - 1)
    
    return args


def scale(profile):
    return (profile[0] / 1e9, profile[1] / 1e6)


def main(args):
    model = MRFSuperNet(args.image_channels, args.num_classes, args.channel_step)
    checkpoint = torch.load(args.ckp_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    
    supernet = model.supernet
    potentials = model._potentials

    solutions = []
    for i in range(args.m):
        logging.info(f"*****Begin inference for solution {i+1}*****")
        gamma = solve_gamma(supernet, potentials, solutions, i)
        mrf = MRF(supernet, gamma=gamma, lams=args.lams[:i], solutions=solutions[:i], potentials=potentials, flops_path=args.flops_path)
        
        solution, flops = inference(mrf)
        solutions.append(solution)

        solution_str = str(solution).replace(" ", "")[3:-3]
        logging.info(f"*****End inference for solution {i+1}*****\n")
        
    image = torch.rand(1, 3, 256, 256)
    for i, solution in enumerate(solutions):
        child = ChildNet(args.image_channels, args.num_classes, args.channel_step, np.array(solution[1:-1]))
        flops, _ = scale(profile(child, inputs=(image, ), verbose=False))
        solution_str = str(solution).replace(" ", "")[3:-3]
        logging.info(f"*****Solution {i+1}*****, FLOPs: {flops:.2f}, solution: {solution_str}")


def inference(mrf): 
    solution = []

    mplp = Mplp(mrf.mrf)
    mplp_query = mplp.map_query()
    
    for (key, value) in mrf.unary.items():
        solution.append(np.where(value == mplp_query[key])[0][0])

    flops = mrf.get_flops(solution)
    solution_str = str(solution).replace(" ", "")[3:-3]
    logging.info(f"FLOPs: {flops:.2f}, solution: {solution_str}")
    
    return solution, flops


def solve_gamma(supernet, potentials, solutions, i):
    gamma_min = args.gamma_min
    gamma_max = args.gamma_max

    iterations = 0
    flops = args.target + 1
    while flops > args.target:
        iterations += 1
        if iterations >= 2:
            logging.info("Too many expanding loops for gamma, try adjusting gamma_max")
        gamma_max *= 2
        mrf = MRF(supernet=supernet, potentials=potentials, gamma=gamma_max, lams=args.lams[:i], solutions=solutions[:i], flops_path=args.flops_path)
        flops = inference(mrf)[1]
        logging.info(f"#iterations: {iterations}, gamma_max: {gamma_max}")
    
    for iter in range(args.gamma_iter):
        gamma_mid = 0.5 * (gamma_min + gamma_max)
        mrf = MRF(supernet=supernet, potentials=potentials, gamma=gamma_mid, lams=args.lams[:i], solutions=solutions[:i], flops_path=args.flops_path)
        flops = inference(mrf)[1]
        if flops > args.target:
            gamma_min = gamma_mid
        else:
            gamma_max = gamma_mid
        logging.info(f"iteration: {iter}, gamma_mid: {gamma_mid}")
    
    return gamma_max


if __name__ == '__main__':
    args = get_args()
    
    # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
 
    logging.basicConfig(filename= "inference.log",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info(str(args))

    main(args)