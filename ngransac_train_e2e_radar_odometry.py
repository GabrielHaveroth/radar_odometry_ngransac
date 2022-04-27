import os
from dataset import RadarCorrespondences
from network import CNNet
import random
import json
import torch
import torch.optim as optim
import ngransac
import argparse
torch.cuda.empty_cache()


arg_parser = argparse.ArgumentParser(description='json config file path')

# Add the arguments
arg_parser.add_argument('config',
                        metavar='config_path',
                        type=str,
                        help='the path to config file')


config_file_path = arg_parser.parse_args().config

with open(config_file_path, 'r') as f:
    configs = json.load(f)


train_data = [sorted([subdir[0] for subdir in os.walk(configs['data_path'])]
                     [1:])[seq] for seq in configs['train_seqs']]

trainset = RadarCorrespondences(train_data, configs['ratio'],
                                configs['nfeatures'], configs['nosideinfo'])

trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                              batch_size=configs['batch_size'])
# create or load model
model = CNNet(configs['resblocks'])
# if len(opt.model) > 0:
# 	model.load_state_dict(torch.load(opt.model))
model = model.cuda()
model.train()

optimizer = optim.Adam(model.parameters(), lr=configs['learningrate'])
train_log = open('log_train.txt', 'w', 1)

iteration = 0
supervised_loss = torch.nn.L1Loss()

# main training loop
for epoch in range(0, configs['epochs']):

    print("=== Starting Epoch", epoch, "==================================")

    # store the network every so often
    torch.save(model.state_dict(), './weights_{}.net'.format(epoch))

    # main training loop in the current epoch
    for correspondences, T12, pts1, pts2 in trainset_loader:
        # predict neural guidance
        log_probs = model(correspondences.float().cuda())
        probs = torch.exp(log_probs).cpu()
        # this tensor will contain the gradients for the entire batch
        log_probs_grad = torch.zeros(log_probs.size())
        avg_loss = 0
        # Loop over batch
        for b in range(correspondences.size(0)):
            # we sample multiple times per input and keep the gradients and losse in the following lists
            log_prob_grads = []
            losses = []
            # loop over samples for approximating the expected loss

            for s in range(configs['samplecount']):
                # gradient tensor of the current sample
                # when running NG-RANSAC, this tensor will indicate which correspondences have been samples
                # this is multiplied with the loss of the sample to yield the gradients for log-probabilities
                gradients = torch.zeros(probs[b].size())
                T12_pred = torch.zeros(T12[b].size()).float()
                # inlier mask of the best model
                inliers = torch.zeros(probs[b].size())
                # random seed used in C++ (would be initialized in each call with the same seed if not provided from outside)
                rand_seed = random.randint(0, 10000)
                # run NG-RANSAC
                incount = ngransac.find_transform(pts2[b].float(), pts1[b].float(), probs[b], rand_seed, 100, float(0.35), T12_pred, gradients)
                incount /= correspondences.size(2)
                # choose the user-defined training signal
                if configs['loss'] == 'inliers':
                    loss = -incount
                else:
                    alpha = 10
                    R_pred = T12_pred[:2, :2]
                    t_pred = T12_pred[:2, 2]
                    R = T12[b][:2, :2].float()
                    t = T12[b][:2, 2].float()
                    t_loss = supervised_loss(t_pred, t)
                    R_loss = supervised_loss(torch.matmul(R_pred.transpose(0, 1), R), torch.eye(2, 2))
                    loss = R_loss + alpha * t_loss

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

        print("Iteration: ", iteration, "Loss: ", avg_loss)

        iteration += 1
