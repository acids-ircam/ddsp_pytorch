#%% -*- coding: utf-8 -*-

# Plotting
import matplotlib
#matplotlib.use('agg')
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import load_dataset

# Default path on my computer
default_path = '/Users/esling/Datasets/instruments_solo_recordings/'
# Define arguments
parser = argparse.ArgumentParser()
# Data arguments
parser.add_argument('--path',           type=str,   default=default_path,   help='Path to the dataset')
parser.add_argument('--output',         type=str,   default='outputs',      help='Output result directory')
parser.add_argument('--dataset',        type=str,   default='violin_simple',help='Name of the dataset')
parser.add_argument('--nbworkers',      type=int,   default=0,              help='Number of parallel workers (multithread)')
# Preprocessing arguments
parser.add_argument('--sr',             type=int,   default=16000,          help='Sample rate of the signal')
parser.add_argument('--f0_estimate',    type=str,   default='crepe',        help='Type of F0 estimate')
parser.add_argument('--fft_scales',     type=list,  default=[64, 6],        help='Minimum and number of scales')
parser.add_argument('--smooth_kernel',  type=int,   default=8,              help='Size of the smoothing kernel')
# DDSP parameters
parser.add_argument('--partials',       type=int,   default=50,             help='Number of partials')
parser.add_argument('--filter_size',    type=int,   default=64,             help='Size of the filter')
parser.add_argument('--block_size',     type=int,   default=160,            help='Size of individual processed blocks')
parser.add_argument('--kernel_size',    type=int,   default=15,             help='Size of the kernel')
parser.add_argument('--sequence_size',  type=int,   default=200,            help='Size of the sequence')
# Model arguments
parser.add_argument('--model',          type=str,   default='vae',          help='Type of model (mlp, cnn, ae, vae, wae, flow)')
parser.add_argument('--layers',         type=str,   default='gru',          help='Type of layers in the model')
parser.add_argument('--strides',        type=list,  default=[2,4,4,5],      help='Set of processing strides')
parser.add_argument('--n_hidden',       type=int,   default=512,            help='Number of hidden units')
parser.add_argument('--n_layers',       type=int,   default=4,              help='Number of computing layers')
parser.add_argument('--channels',       type=int,   default=128,            help='Number of channels in convolution')
parser.add_argument('--kernel',         type=int,   default=15,             help='Size of convolution kernel')
parser.add_argument('--encoder_dims',   type=int,   default=64,             help='Number of encoder output dimensions')
parser.add_argument('--latent_dims',    type=int,   default=0,              help='Number of latent dimensions')
parser.add_argument('--warm_amp',       type=int,   default=0,              help='Warmup on amplitude training')
parser.add_argument('--warm_synth',     type=int,   default=200,            help='Warmup on synthesis')
parser.add_argument('--warm_noise',     type=int,   default=2000,           help='Warmup on noise')
parser.add_argument('--beta_factor',    type=int,   default=1,              help='Beta factor in VAE')
# Flow specific parameters
parser.add_argument('--flow',           type=str,   default='iaf',          help='Type of flow to use')
parser.add_argument('--flow_length',    type=int,   default=8,              help='Number of flow transforms')
# Optimization arguments
parser.add_argument('--early_stop',     type=int,   default=60,             help='Early stopping')
parser.add_argument('--train_type',     type=str,   default='fixed',        help='Fixed or random data split')
parser.add_argument('--plot_interval',  type=int,   default=100,            help='Interval of plotting frequency')
parser.add_argument('--batch_size',     type=int,   default=64,             help='Size of the batch')
parser.add_argument('--epochs',         type=int,   default=200,            help='Number of epochs to train on')
parser.add_argument('--lr',             type=float, default=2e-4,           help='Learning rate')
# Evaluation parameters
parser.add_argument('--check_exists',   type=int,   default=0,              help='Check if model exists')
parser.add_argument('--time_limit',     type=int,   default=0,              help='Maximum time to train (in minutes)')
# CUDA arguments
parser.add_argument('--device',         type=str,   default='cpu',          help='Device for CUDA')
args = parser.parse_args()
# Track start time (for HPC)
start_time = time.time()
if (args.device != 'cpu'):
    # Enable CuDNN optimization
    torch.backends.cudnn.benchmark=True

#%%
"""
###################
Basic definitions
###################
"""
# Results and checkpoint folders
if not os.path.exists('{0}'.format(args.output)):
    os.makedirs('{0}'.format(args.output))
    os.makedirs('{0}/audio'.format(args.output))
    os.makedirs('{0}/images'.format(args.output))
    os.makedirs('{0}/models'.format(args.output))
# Model save file
model_name = '{0}_{1}'.format(args.model, str(args.latent_dims))
base_dir = '{0}/'.format(args.output)
base_img = '{0}/images/{1}'.format(args.output, model_name)
base_audio = '{0}/audio/{1}'.format(args.output, model_name)
if (args.check_exists == 1):
    if os.path.exists(args.output + '/models/' + model_name + '.th'):
        print('[Found ' + args.output + '/models/' + model_name + '.th - Exiting.]')
        exit
# Handling cuda
args.cuda = not args.device == 'cpu' and torch.cuda.is_available()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('Optimization will be on ' + str(args.device) + '.')

"""
###################
Dataset import definitions
################### 
"""
print('[Loading dataset]')
ref_split = args.path + '/reference_split_' + args.dataset + '.th'
if (args.train_type == 'random' or (not os.path.exists(ref_split))):
    train_loader, valid_loader, test_loader, args = load_dataset(args)
    if (args.train_type == 'fixed'):
        torch.save([train_loader, valid_loader, test_loader], ref_split)
else:
    data = torch.load(ref_split)
    train_loader, valid_loader, test_loader = data[0], data[1], data[2]
    args.output_size = train_loader.dataset.output_size
    args.input_size = train_loader.dataset.input_size
# Take fixed batch for plot purposes
fixed_data, fixed_params, fixed_meta, fixed_audio = next(iter(test_loader))
fixed_data, fixed_params, fixed_meta, fixed_audio = fixed_data.to(args.device), fixed_params.to(args.device), fixed_meta, fixed_audio
fixed_batch = (fixed_data, fixed_params, fixed_meta, fixed_audio)
# Set latent dims to output dims
if (args.latent_dims == 0):
    args.latent_dims = args.output_size

"""
###################
Model definition section
###################
"""
print('[Creating model]')
if (model in ['ae', 'vae', 'wae']):
    model = NeuralSynth()
else:
    raise Exception('Unknown model ' + args.model)
# Send model to device
model = model.to(args.device)

"""
###################
Optimizer section
###################
"""
# Optimizer model
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True, threshold=1e-7)
# Loss
if (args.loss == 'mse'):
    loss = nn.MSELoss(reduction='mean').to(args.device)
elif (args.loss == 'l1'):
    loss = nn.SmoothL1Loss(reduction='mean').to(args.device)
elif (args.loss == 'bce'):
    loss = nn.BCELoss(reduction='mean').to(args.device)
elif (args.loss == 'multinomial'):
    loss = multinomial_loss
elif (args.loss == 'multi_mse'):
    loss = multinomial_mse_loss
else:
    raise Exception('Unknown loss ' + args.loss)

"""
###################
Training section
###################
"""
#% Monitoring quantities
losses = torch.zeros(args.epochs, 3)
if (args.epochs == 0):
    losses = torch.zeros(200, 3)
best_loss = np.inf
early = 0
print('[Starting training]')
for i in range(args.epochs):
    if (args.start_regress == 0):
        from pympler import muppy, summary
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        # Prints out a summary of the large objects
        print('************ Summary at beginning of epoch ************')
        summary.print_(sum1)
    # Set warm-up values
    args.beta = args.beta_factor * (float(i) / float(max(args.warm_latent, i)))
    if (i >= args.start_regress):
        args.gamma = ((float(i - args.start_regress) * args.reg_factor) / float(max(args.warm_regress, i - args.start_regress)))
        if (args.regressor != 'mlp'):
            args.gamma *= 1e-1
    else:
        args.gamma = 0
    if (i >= args.start_disentangle):
        args.delta = ((float(i - args.start_disentangle)) / float(max(args.warm_disentangle, i - args.start_disentangle)))
    else:
        args.delta = 0
    print('%.3f - %.3f'%(args.beta, args.gamma))
    # Perform one epoch of train
    losses[i, 0] = model.train_epoch(train_loader, loss, optimizer, args)    
    # Perform validation
    losses[i, 1] = model.eval_epoch(valid_loader, loss, args)
    # Learning rate scheduling
    if ((not args.model in ['ae', 'vae', 'wae', 'vae_flow']) or (i >= args.start_regress)):
        scheduler.step(losses[i, 1])
    # Perform test evaluation
    losses[i, 2] = model.eval_epoch(test_loader, loss, args)
    if (args.start_regress == 1000):
        losses[i, 1] = losses[i, 0]
        losses[i, 2] = losses[i, 0]
    # Model saving
    if (losses[i, 1] < best_loss):
        # Save model
        best_loss = losses[i, 1]
        torch.save(model, args.output + '/models/' + model_name + '.model')
        early = 0
    # Check for early stopping
    elif (args.early_stop > 0 and i >= args.start_regress):
        early += 1
        if (early > args.early_stop):
            print('[Model stopped early]')
            break
    # Periodic evaluation (or debug model)
    if ((i + 1) % args.plot_interval == 0 or (args.epochs == 1)):
        args.plot = 'train'
        with torch.no_grad():
            model.eval()
            evaluate_model(model, fixed_batch, test_loader, args, train=True, name=base_img + '_batch_' + str(i))
    # Time limit for HPC grid eval
    if ((args.time_limit > 0) and (((time.time() - start_time) / 60.0) > args.time_limit)):
        print('[Hitting time limit after ' + str((time.time() - start_time) / 60.0) + ' minutes.]')
        print('[Going to evaluation mode]')
        break
    if (args.regressor == 'flow_kl_f'):
        print(torch.cuda.memory_allocated(args.device))
    print('Epoch ' + str(i))
    print(losses[i])

"""
###################
Evaluation section
###################
"""
from evaluate import evaluate_params, evaluate_synthesis, evaluate_projection
from evaluate import evaluate_reconstruction, evaluate_latent_space
from evaluate import evaluate_meta_parameters, evaluate_semantic_parameters
from evaluate import evaluate_latent_neighborhood
args.plot = 'final'
args.model_name, args.base_img, args.base_audio = model_name, base_img, base_audio
args.base_model = args.output + '/models/' + model_name
print('[Reload best performing model]')
model = torch.load(args.output + '/models/' + model_name + '.model')
model = model.to(args.device)
print('[Performing final evaluation]')
# Memory saver
with torch.no_grad():
    # Perform parameters evaluation
    evaluate_params(model, test_loader, args, losses=losses)
    # Synthesis engine (on GPU)
    if (args.synthesize):
        # Import synthesis
        from synth.synthesize import create_synth
        print('[Synthesis evaluation]')
        # Create synth rendering system
        args.engine, args.generator, args.param_defaults, args.rev_idx = create_synth(args.dataset)
    # Evaluation specific to AE models
    if (args.model not in ['mlp', 'gated_mlp', 'cnn', 'gated_cnn', 'res_cnn']):
        # Perform reconstruction evaluation
        evaluate_reconstruction(model, test_loader, args, train=False)
        # Evaluate latent space
        args = evaluate_latent_space(model, test_loader, args, train=False)
        # Perform meta-parameter analysis
        evaluate_meta_parameters(model, test_loader, args, train=False)
        # Perform latent neighborhood analysis
        evaluate_latent_neighborhood(model, test_loader, args, train=False)
        # Perform semantic parameter analysis
        evaluate_semantic_parameters(model, test_loader, args, train=False)
    # Synthesis engine (on GPU)
    if (args.synthesize):
        # Evaluate synthesizer output
        evaluate_synthesis(model, test_loader, args, train=False)
        print('[Load set of testing sound (outside Diva)]')
        test_sounds = get_external_sounds(args.test_sounds, test_loader.dataset, args)
        # Evaluate projection
        evaluate_projection(model, test_sounds, args, train=False)
        print('[Evaluate vocal sketching dataset]')
        test_sounds = get_external_sounds(args.vocal_sounds, test_loader.dataset, args)
        # Evaluate projection
        evaluate_projection(model, test_sounds, args, train=False, type_val='vocal')
