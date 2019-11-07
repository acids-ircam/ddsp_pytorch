#%% -*- coding: utf-8 -*-

# Plotting
import matplotlib
matplotlib.use('agg')
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Internal imports
from utils.data import load_dataset, get_external_sounds
from models.vae.ae import AE, RegressionAE, DisentanglingAE
from models.vae.vae import VAE
from models.vae.wae import WAE
from models.vae.vae_flow import VAEFlow
from models.loss import multinomial_loss, multinomial_mse_loss
from models.basic import GatedMLP, GatedCNN, construct_encoder_decoder, construct_flow, construct_regressor, construct_disentangle
from evaluate import evaluate_model

# Define arguments
parser = argparse.ArgumentParser()
# Data arguments
parser.add_argument('--path',           type=str,   default='',             help='')
parser.add_argument('--test_sounds',    type=str,   default='',             help='')
parser.add_argument('--output',         type=str,   default='outputs',      help='')
parser.add_argument('--dataset',        type=str,   default='32par',        help='')
parser.add_argument('--train_type',     type=str,   default='fixed',        help='')
parser.add_argument('--nbworkers',      type=int,   default=0,              help='')
# Model arguments
parser.add_argument('--model',          type=str,   default='vae',          help='')
parser.add_argument('--loss',           type=str,   default='mse',          help='')
parser.add_argument('--n_classes',      type=int,   default=61,             help='')
parser.add_argument('--n_hidden',       type=int,   default=1024,           help='')
parser.add_argument('--n_layers',       type=int,   default=4,              help='')
# CNN parameters
parser.add_argument('--channels',       type=int,   default=64,             help='')
parser.add_argument('--kernel',         type=int,   default=5,              help='')
parser.add_argument('--dilation',       type=int,   default=3,              help='')
# AE-specific parameters
parser.add_argument('--layers',         type=str,   default='gated_cnn',    help='')
parser.add_argument('--encoder_dims',   type=int,   default=64,             help='')
parser.add_argument('--latent_dims',    type=int,   default=0,              help='')
parser.add_argument('--warm_latent',    type=int,   default=50,             help='')
parser.add_argument('--beta_factor',    type=int,   default=1,              help='')
# Two-step training parameters
parser.add_argument('--ref_model',      type=str,   default='',             help='')
# Flow specific parameters
parser.add_argument('--flow',           type=str,   default='iaf',          help='')
parser.add_argument('--flow_length',    type=int,   default=8,              help='')
# Optimization arguments
parser.add_argument('--k_run',          type=int,   default=0,              help='')
parser.add_argument('--early_stop',     type=int,   default=60,             help='')
parser.add_argument('--plot_interval',  type=int,   default=100,            help='')
parser.add_argument('--batch_size',     type=int,   default=64,             help='')
parser.add_argument('--epochs',         type=int,   default=200,            help='')
parser.add_argument('--eval',           type=int,   default=100,            help='')
parser.add_argument('--lr',             type=float, default=2e-4,           help='')
# Evaluation parameters
parser.add_argument('--batch_evals',    type=int,   default=16,             help='')
parser.add_argument('--batch_out',      type=int,   default=3,              help='')
parser.add_argument('--check_exists',   type=int,   default=0,              help='')
parser.add_argument('--time_limit',     type=int,   default=0,              help='')
# CUDA arguments
parser.add_argument('--device',         type=str,   default='cpu',          help='Device for CUDA')
args = parser.parse_args()
# Track start time (for HPC)
start_time = time.time()
# In case we are CPU
args.synthesize = False
# Parameter checking
if (len(args.path) == 0):
    args.path = (args.device == 'cpu') and '/Users/esling/Datasets/diva_dataset' or '/fast-2/datasets/diva_dataset/'
    args.test_sounds = (args.device == 'cpu') and '/Users/esling/Datasets/synth_testing' or '/fast-2/datasets/flow_synthesizer/synth_testing'
    args.vocal_sounds = '/fast-2/datasets/flow_synthesizer/vocal_testing'
    #args.output = (args.device == 'cpu') and 'outputs' or '/fast-1/philippe/flow_results'
if (args.device not in ['cpu']):
    args.synthesize = True
if (args.device != 'cpu'):
    # Enable CuDNN optimization
    torch.backends.cudnn.benchmark=True

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
model_name = '{0}_{1}_{2}_{3}'.format(args.model, args.data, args.loss, str(args.latent_dims))
if (not (args.model in ['mlp', 'gated_mlp', 'cnn', 'gated_cnn', 'res_cnn'])):
    model_name += '_' + args.layers
    if (args.model == 'vae_flow'):
        model_name += '_' + args.flow
    model_name += '_' + args.regressor
    if (args.regressor != 'mlp'):
        model_name += '_' + args.reg_flow + '_' + str(args.reg_layers)
    if (args.semantic_dim > -1):
        model_name += '_' + str(args.semantic_dim) + '_' + args.disentangling
if (args.k_run > 0):
    model_name += '_' + str(args.k_run)
base_dir = '{0}/'.format(args.output)
base_img = '{0}/images/{1}'.format(args.output, model_name)
base_audio = '{0}/audio/{1}'.format(args.output, model_name)
if (args.check_exists == 1):
    if os.path.exists(args.output + '/models/' + model_name + '.synth.results.npy'):
        print('[Found ' + args.output + '/models/' + model_name + '.synth.results.npy - Exiting.]')
        exit
# Handling cuda
args.cuda = not args.device == 'cpu' and torch.cuda.is_available()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('Optimization will be on ' + str(args.device) + '.')

"""
###################
Basic definitions
################### 
"""
print('[Loading dataset]')
ref_split = args.path + '/reference_split_' + args.dataset+ "_" + args.data + '.th'
if (args.train_type == 'random' or (not os.path.exists(ref_split))):
    train_loader, valid_loader, test_loader, args = load_dataset(args)
    if (args.train_type == 'fixed'):
        torch.save([train_loader, valid_loader, test_loader], ref_split)
    # Take fixed batch
    fixed_data, fixed_params, fixed_meta, fixed_audio = next(iter(test_loader))
    fixed_data, fixed_params, fixed_meta, fixed_audio = fixed_data.to(args.device), fixed_params.to(args.device), fixed_meta, fixed_audio
    fixed_batch = (fixed_data, fixed_params, fixed_meta, fixed_audio)
else:
    data = torch.load(ref_split)
    train_loader, valid_loader, test_loader = data[0], data[1], data[2]
    fixed_data, fixed_params, fixed_meta, fixed_audio = next(iter(test_loader))
    fixed_data, fixed_params, fixed_meta, fixed_audio = fixed_data.to(args.device), fixed_params.to(args.device), fixed_meta, fixed_audio
    fixed_batch = (fixed_data, fixed_params, fixed_meta, fixed_audio)
    args.output_size = train_loader.dataset.output_size
    args.input_size = train_loader.dataset.input_size
# Set latent dims to output dims
if (args.latent_dims == 0):
    args.latent_dims = args.output_size

"""
###################
Model definition section
###################
"""
print('[Creating model]')
if (args.loss in ['multinomial']):
    args.output_size *= args.n_classes
if (args.loss in ['multi_mse']):
    args.output_size *= (args.n_classes + 1)
if (args.model == 'mlp'):
    model = GatedMLP(np.prod(args.input_size), args.output_size, hidden_size = args.n_hidden, n_layers = args.n_layers, type_mod='normal')
elif (args.model == 'gated_mlp'): 
    model = GatedMLP(np.prod(args.input_size), args.output_size, hidden_size = args.n_hidden, n_layers = args.n_layers, type_mod='gated')
elif (args.model == 'cnn'):
    model = GatedCNN(args.input_size, args.output_size, channels = args.channels, n_layers = 4, hidden_size = args.n_hidden, n_mlp = 3, type_mod='normal', args=args)
elif (args.model == 'gated_cnn'):
    model = GatedCNN(args.input_size, args.output_size, channels = args.channels, n_layers = 4, hidden_size = args.n_hidden, n_mlp = 3, type_mod='gated', args=args)
elif (args.model == 'res_cnn'):
    model = GatedCNN(args.input_size, args.output_size, channels = args.channels, n_layers = 4, hidden_size = args.n_hidden, n_mlp = 3, type_mod='residual', args=args)
elif (args.model in ['ae', 'vae', 'wae', 'vae_flow']):
    # Construct reconstruction loss
    if (args.rec_loss == 'mse'):
        rec_loss = nn.MSELoss(reduction='sum').to(args.device)
    elif (args.rec_loss == 'l1'):
        rec_loss = nn.SmoothL1Loss(reduction='sum').to(args.device)
    elif (args.rec_loss == 'multinomial'):
        rec_loss = multinomial_loss
    elif (args.rec_loss == 'multi_mse'):
        rec_loss = multinomial_mse_loss
    else:
        raise Exception('Unknown reconstruction loss ' + args.rec_loss)
    # Construct encoder and decoder
    encoder, decoder = construct_encoder_decoder(args.input_size, args.encoder_dims, args.latent_dims, channels = args.channels, n_layers = args.n_layers, hidden_size = args.n_hidden, n_mlp = args.n_layers // 2, type_mod=args.layers, args=args)
    # Construct specific type of AE
    if (args.model == 'ae'):
        model = AE(encoder, decoder, args.encoder_dims, args.latent_dims)
    elif (args.model == 'vae'):
        model = VAE(encoder, decoder, args.input_size, args.encoder_dims, args.latent_dims)
    elif (args.model == 'wae'):
        model = WAE(encoder, decoder, args.input_size, args.encoder_dims, args.latent_dims)
    elif (args.model == 'vae_flow'):
        # Construct the normalizing flow
        flow, blocks = construct_flow(args.latent_dims, flow_type=args.flow, flow_length=args.flow_length, amortization='input')
        # Construct full VAE with given flow
        model = VAEFlow(encoder, decoder, flow, args.input_size, args.encoder_dims, args.latent_dims)
    # Construct specific regressor
    regression_model = construct_regressor(args.latent_dims, args.output_size, model=args.regressor, hidden_dims = args.reg_hiddens, n_layers=args.reg_layers, flow_type=args.reg_flow)
    if (args.semantic_dim == -1):
        # Final AE / Regression model
        model = RegressionAE(model, args.latent_dims, args.output_size, rec_loss, regressor=regression_model, regressor_name=args.regressor)
    else:
        # Construct disentangling flow
        disentangling = construct_disentangle(args.latent_dims, model=args.disentangling, semantic_dim=args.semantic_dim, n_layers=args.dis_layers, flow_type=args.reg_flow)
        # Final AE / Disentanglement / Regression model
        model = DisentanglingAE(model, args.latent_dims, args.output_size, rec_loss, regressor=regression_model, regressor_name=args.regressor, disentangling=disentangling, semantic_dim=args.semantic_dim)
else:
    raise Exception('Unknown model ' + args.model)
# Send model to device
model = model.to(args.device)

# Two-step training loading procedure
if (len(args.ref_model) > 0):
    print('[Loading reference ' + args.ref_model + ']')
    ref_model = torch.load(args.ref_model)#, map_location=args.device)
    if (args.regressor != 'mlp'):
        ref_model_ae = ref_model.ae_model.to(args.device)
        model.ae_model = None
        model.ae_model = ref_model_ae
        ref_model = None
    else:
        model = None
        model = ref_model.to(args.device)

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
