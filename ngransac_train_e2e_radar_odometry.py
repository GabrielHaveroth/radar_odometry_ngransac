import os
from dataset import RadarCorrespondences, RadarCorrespondencesHdf5
from network import CNNet
import random
import json
import torch
import torch.optim as optim
import ngransac
import argparse
import wandb
from datetime import datetime

torch.cuda.empty_cache()
random.seed(1)


def val_loop(valset_loader, model, loss):
    print("===== Validate NG-RANSAC Odometry =====")
    val_losses = []
    model.eval()
    for correspondences, T12, pts1, pts2 in valset_loader:
        # Predict neural guidance
        log_probs = model(correspondences.float().cuda())
        probs = torch.exp(log_probs).cpu()
        with torch.no_grad():
            for b in range(correspondences.size(0)):
                gradients = torch.zeros(probs[b].size())
                T12_pred = torch.zeros(T12[b].size()).float()
                # random seed used in C++ (would be initialized in each call
                # with the same seed if not provided from outside)
                rand_seed = random.randint(0, 10000)
                incount = ngransac.find_transform(pts2[b].float(),
                                                  pts1[b].float(),
                                                  probs[b], rand_seed,
                                                  1000, float(0.01),
                                                  T12_pred, gradients)
                incount /= correspondences.size(2)
                alpha = 10
                R_pred = T12_pred[:2, :2]
                t_pred = T12_pred[:2, 2]
                R = T12[b][:2, :2].float()
                t = T12[b][:2, 2].float()
                t_loss = loss(t_pred, t)
                R_loss = loss(torch.matmul(
                              R_pred.transpose(0, 1), R),
                              torch.eye(2, 2))
                val_loss = t_loss + alpha * R_loss
                val_losses.append(val_loss)

    val_loss_avg = sum(val_losses) / len(val_losses)
    print(f"Avg loss: {val_loss_avg:>8f} \n")
    random.seed()
    return val_loss_avg


arg_parser = argparse.ArgumentParser(description='json config file path')

# Add the arguments
arg_parser.add_argument('config',
                        metavar='config_path',
                        type=str,
                        help='the path to config file')


config_file_path = arg_parser.parse_args().config

with open(config_file_path, 'r') as f:
    configs = json.load(f)

if configs['use_hdf5']:
    trainset = RadarCorrespondencesHdf5(configs['hdf5_path_train'],
                                        configs['ratio'],
                                        configs['nfeatures'],
                                        configs['nosideinfo'])
    valset = RadarCorrespondencesHdf5(configs['hdf5_path_val'],
                                      configs['ratio'],
                                      configs['nfeatures'],
                                      configs['nosideinfo'])
else:
    total_data = sorted([subdir[0]
                         for subdir in os.walk(configs['data_path'])][1:])
    train_data = [total_data[seq] for seq in configs['train_seqs']]
    val_data = [total_data[seq] for seq in configs['val_seqs']]

    trainset = RadarCorrespondences(train_data, configs['ratio'],
                                    configs['nfeatures'], configs['nosideinfo'])

    valset = RadarCorrespondences(val_data, configs['ratio'],
                                  configs['nfeatures'], configs['nosideinfo'])

valset_loader = torch.utils.data.DataLoader(valset, shuffle=False,
                                            batch_size=configs['batch_size'], num_workers=configs['number_workers'])

trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                              batch_size=configs['batch_size'], num_workers=configs['number_workers'])
# create or load model
model = CNNet(configs['resblocks'])
# if len(opt.model) > 0:
# 	model.load_state_dict(torch.load(opt.model))
model = model.cuda()

check_point = {}
STEPS_TO_VALIDATE = configs['steps_to_validate']
STEPS_TO_SAVE_CHECKPOINT = configs['steps_to_save_checkpoint']
STEPS_TO_PRINT_RESULTS = configs['steps_to_print_results']

optimizer = optim.Adam(model.parameters(), lr=configs['learningrate'])

supervised_loss = torch.nn.L1Loss()
length_train = len(trainset_loader)
length_val = len(valset_loader)
print('train pairs per batch {}'.format(length_train))
print('val pairs per batch {}'.format(length_val))
actual_datatime = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

if configs['log_wandb']:
    wandb.init(project=configs['wandb_project_name'],
               name=f'experiment_{actual_datatime}')
# Main training loop
for epoch in range(0, configs['epochs']):
    train_log = open(f'log_train{epoch + 1}.txt', 'w', 1)
    iteration = 1
    print("=== Starting Epoch", epoch + 1,
          "==================================")

    # store the network every so often
    model.train()
    # Main training loop in the current epoch
    for correspondences, T12, pts1, pts2 in trainset_loader:
        # Predict neural guidance
        log_probs = model(correspondences.cuda())
        probs = torch.exp(log_probs).cpu()
        # this tensor will contain the gradients for the entire batch
        log_probs_grad = torch.zeros(log_probs.size())
        avg_loss = 0
        # Loop over batch
        for b in range(correspondences.size(0)):
            # we sample multiple times per input and keep the gradients
            # and losse in the following lists
            log_prob_grads = []
            losses = []
            # loop over samples for approximating the expected loss

            for s in range(configs['samplecount']):
                # gradient tensor of the current sample
                # when running NG-RANSAC, this tensor will indicate which correspondences have been samples
                # this is multiplied with the loss of the sample to yield the gradients for log-probabilities
                gradients = torch.zeros(probs[b].size())
                T12_pred = torch.zeros(T12[b].size()).float()
                # random seed used in C++ (would be initialized in each call with the same seed if not provided from outside)
                rand_seed = random.randint(0, 10000)
                # run NG-RANSAC
                incount = ngransac.find_transform(pts2[b].float(),
                                                  pts1[b].float(),
                                                  probs[b],
                                                  rand_seed,
                                                  1000,
                                                  float(0.35),
                                                  T12_pred,
                                                  gradients)
                incount /= correspondences.size(2)
                # choose the user-defined training signal
                if configs['loss'] == 'inliers':
                    loss = -incount
                else:
                    alpha = configs['alpha']
                    R_pred = T12_pred[:2, :2]
                    t_pred = T12_pred[:2, 2]
                    R = T12[b][:2, :2].float()
                    t = T12[b][:2, 2].float()
                    t_loss = supervised_loss(t_pred, t)
                    R_loss = supervised_loss(torch.matmul(
                                             R_pred.transpose(0, 1), R),
                                             torch.eye(2, 2))
                    loss = t_loss + alpha * R_loss

                log_prob_grads.append(gradients)
                losses.append(loss)
            # calculate the gradients of the expected loss
            baseline = sum(losses) / len(losses)  # expected loss
            # substract baseline for each sample to reduce gradient variance
            for i, l in enumerate(losses):
                log_probs_grad[b] += log_prob_grads[i] * \
                    (l - baseline) / configs['samplecount']

            avg_loss += baseline

        avg_loss /= correspondences.size(0)
        train_log.write('%d %f\n' % (iteration, avg_loss))

        # update model
        torch.autograd.backward((log_probs), (log_probs_grad.cuda()))
        optimizer.step()
        optimizer.zero_grad()

        if iteration % STEPS_TO_PRINT_RESULTS == 0:
            current = iteration
            metrics = {"train/loss": avg_loss}
            if configs['log_wandb']:
                wandb.log(metrics)
            print(f"loss: {avg_loss:>7f}  [{current:>5d}/{length_train:>5d}]")

        if iteration % STEPS_TO_SAVE_CHECKPOINT == 0:
            actual_datatime = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            torch.save({'step': iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss},
                       configs['save_models_path'] + f'model_step_{iteration}_epoch_{epoch}_{actual_datatime}.pt')

        if iteration % STEPS_TO_VALIDATE == 0:
            avg_val_loss = val_loop(valset_loader, model, supervised_loss)
            metrics = {"val/loss": avg_val_loss}
            model.train()
            if configs['log_wandb']:
                wandb.log(metrics)
        iteration += 1
