import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from models import GnnNets
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args, mcts_args
from my_mcts import mcts
from tqdm import tqdm
from proto_join import join_prototypes_by_activations
from utils import PlotUtils
from torch_geometric.utils import to_networkx
from itertools import accumulate
from torch_geometric.datasets import MoleculeNet
import pdb
import random


def warm_only(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def append_record(info):
    with open('./log/hyper_search.txt', 'a') as f:
        f.write(info + '\n')


# train for graph classification
def train_GC(model_type):

    print('start loading data====================')
    dataset_tuple = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)

    # If the dataset is a tuple (train, val, test)
    if isinstance(dataset_tuple, tuple):
        train_dataset, val_dataset, test_dataset = dataset_tuple
        input_dim = train_dataset[0].x.shape[1]
        output_dim = int(max([data.y.item() for data in train_dataset]) + 1)
    else:
        dataset = dataset_tuple
        input_dim = dataset.num_node_features
        output_dim = int(dataset.num_classes)
        dataloader = get_dataloader(dataset, data_args.dataset_name, train_args.batch_size,
                                    data_split_ratio=data_args.data_split_ratio)

    # dataloaders
    if isinstance(dataset_tuple, tuple):
        train_loader = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=train_args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=train_args.batch_size, shuffle=False)
        dataset_for_stats = train_dataset
        data_indices = list(range(len(train_dataset)))
    else:
        train_loader = dataloader['train']
        val_loader   = dataloader['eval']
        test_loader  = dataloader['test']
        dataset_for_stats = dataset
        data_indices = dataloader['train'].dataset.indices

    print('start training model==================')

    gnnNets = GnnNets(input_dim, output_dim, model_args)
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    # compute dataset stats
    avg_nodes = sum([data.x.shape[0] for data in dataset_for_stats]) / len(dataset_for_stats)
    avg_edge_index = sum([data.edge_index.shape[1] for data in dataset_for_stats]) / len(dataset_for_stats)
    print("Dataset : ", data_args.dataset_name)
    print(f"graphs {len(dataset_for_stats)}, avg_nodes {avg_nodes:.4f}, avg_edge_index {avg_edge_index/2:.4f}")

    # make checkpoint directories
    os.makedirs(ckpt_dir, exist_ok=True)

    early_stop_count = 0
    best_acc = 0.0

    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        ld_loss_list = []

        if epoch >= train_args.proj_epochs and epoch % 50 == 0:
            gnnNets.eval()
            # prototype projection
            for i in range(gnnNets.model.prototype_vectors.shape[0]):
                count = 0
                best_similarity = 0
                label = gnnNets.model.prototype_class_identity[0].max(0)[1]
                for j in range(i*10, len(data_indices)):
                    data = dataset_for_stats[data_indices[j]]
                    if data.y == label:
                        count += 1
                        coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i])
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= train_args.count:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        print('Projection of prototype completed')
                        break

            # prototype merge
            if train_args.share:
                if gnnNets.model.prototype_vectors.shape[0] > round(output_dim * model_args.num_prototypes_per_class * (1-train_args.merge_p)):
                    join_info = join_prototypes_by_activations(gnnNets, train_args.proto_percnetile, train_loader, optimizer)

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)

        for batch in train_loader:
            if model_args.cont:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = gnnNets(batch)
            else:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = gnnNets(batch)

            loss = criterion(logits, batch.y)

            if model_args.cont:
                prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y]).to(model_args.device)
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                positive_sim_matrix = sim_matrix * prototypes_of_correct_class
                negative_sim_matrix = sim_matrix * prototypes_of_wrong_class
                contrastive_loss = (positive_sim_matrix.sum(dim=1)) / (negative_sim_matrix.sum(dim=1))
                contrastive_loss = - torch.log(contrastive_loss).mean()

            # diversity loss
            prototype_numbers = [int(torch.count_nonzero(gnnNets.model.prototype_class_identity[:, i])) for i in range(gnnNets.model.prototype_class_identity.shape[1])]
            prototype_numbers = list(accumulate(prototype_numbers))
            n = 0
            ld = 0
            for k in prototype_numbers:
                p = gnnNets.model.prototype_vectors[n:k]
                n = k
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3
                matrix2 = torch.zeros(matrix1.shape).to(model_args.device)
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

            if model_args.cont:
                loss = loss + train_args.alpha2 * contrastive_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss
            else:
                loss = loss + train_args.alpha2 * prototype_pred_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        # report train msg
        print(f"Train Epoch:{epoch} | Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | Acc: {np.concatenate(acc, axis=0).mean():.3f}")
        append_record(f"Epoch {epoch:2d}, loss: {np.average(loss_list):.3f}, acc: {np.concatenate(acc, axis=0).mean():.3f}")

        # evaluation
        eval_state = evaluate_GC(val_loader, gnnNets, criterion)
        print(f"Eval Epoch:{epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")
        append_record(f"Eval epoch {epoch:2d}, loss: {eval_state['loss']:.3f}, acc: {eval_state['acc']:.3f}")

        test_state, _, _ = test_GC(test_loader, gnnNets, criterion)
        print(f"Test Epoch:{epoch} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")

        is_best = (eval_state['acc'] > best_acc)
        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best)

    print(f"The best validation accuracy is {best_acc}.")

    # final test
    gnnNets = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_{model_type}_{model_args.readout}_best.pth'))
    gnnNets.to_device()
    test_state, _, _ = test_GC(test_loader, gnnNets, criterion)
    print(f"Test | Dataset: {data_args.dataset_name} | model: {model_args.model_name}_{model_type} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
    append_record(f"loss: {test_state['loss']:.3f}, acc: {test_state['acc']:.3f}")

    return test_state['acc']


def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, _, _, _, _, _, _ = gnnNets(batch)
            if data_args.dataset_name == 'clintox':
                batch.y = torch.tensor([torch.argmax(i).item() for i in batch.y]).to(model_args.device)
            loss = criterion(logits, batch.y)
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

    return {'loss': np.average(loss_list), 'acc': np.concatenate(acc, axis=0).mean()}


def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _, _, _, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

    test_state = {'loss': np.average(loss_list), 'acc': np.average(np.concatenate(acc, axis=0).mean())}
    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    gnnNets.to('cpu')
    state = {'net': gnnNets.state_dict(), 'epoch': epoch, 'acc': eval_acc}
    pth_name = f"{model_name}_{model_type}_{model_args.readout}_latest.pth"
    best_pth_name = f'{model_name}_{model_type}_{model_args.readout}_best.pth'
    torch.save(state, os.path.join(ckpt_dir, pth_name))
    if is_best:
        torch.save(gnnNets, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to(model_args.device)


if __name__ == '__main__':
    if os.path.isfile("./log/hyper_search.txt"):
        os.remove("./log/hyper_search.txt")

    model_type = 'cont' if model_args.cont else 'var'
    accuracy = train_GC(model_type)
