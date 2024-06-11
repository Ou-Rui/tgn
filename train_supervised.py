'''
python train_supervised.py --use_memory --prefix tgn-attn --n_runs 10 --use_validation -d txn_filter
'''
import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_node_classification

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')
parser.add_argument('--new_node', action='store_true', help='model new node')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + '\
  node-classification.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}' + '\
  node-classification.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

def save_embs(epoch, emb_tuple, prob_tuple, h1_tuple, h2_tuple):
  emb_l, prob_l, h1_l, h2_l = [], [], [], []
  for emb_mode, prob_mode, h1_mode, h2_mode in zip(
      emb_tuple, prob_tuple, h1_tuple, h2_tuple):
    emb_l.extend(emb_mode)
    prob_l.extend(prob_mode)
    h1_l.extend(h1_mode)
    h2_l.extend(h2_mode)
  emb_l = np.array(emb_l)
  prob_l = np.array(prob_l)
  h1_l = np.array(h1_l)
  h2_l = np.array(h2_l)
  np.save(f"./saved_embs/tgn_{args.prefix}_{args.data}_epoch{epoch}_embs.npy", emb_l)
  np.save(f"./saved_embs/tgn_{args.prefix}_{args.data}_epoch{epoch}_probs.npy", prob_l)
  np.save(f"./saved_embs/tgn_{args.prefix}_{args.data}_epoch{epoch}_h1.npy", h1_l)
  np.save(f"./saved_embs/tgn_{args.prefix}_{args.data}_epoch{epoch}_h2.npy", h2_l)

full_data, node_features, edge_features, train_data, val_data, test_data = \
  get_data_node_classification(DATA, use_validation=args.use_validation)

max_idx = max(full_data.unique_nodes)

train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)


# Run Loop
METRICS = ['AUC', 'AP', 'F1', 'RECALL']
best_epoch_l = []
best_val_metric_l = []
best_test_metric_l = []
max_test_metric_l = []
for i in range(args.n_runs):
  results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                i) if i > 0 else "results/{}_node_classification.pkl".format(
    args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message)

  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)
  
  logger.info('Num of training instances: {}'.format(num_instance))
  logger.info('Num of batches per epoch: {}'.format(num_batch))

  logger.info('Loading saved TGN model')
  model_path = f'./saved_models/{args.prefix}-{DATA}.pth'
  tgn.load_state_dict(torch.load(model_path))
  tgn.eval()
  logger.info('TGN models loaded')
  logger.info('Start training node classification task')

  decoder = MLP(node_features.shape[1], drop=DROP_OUT)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
  decoder = decoder.to(device)
  decoder_loss_criterion = torch.nn.BCELoss()

  val_auc_l = []
  train_loss_l = []
  max_val_auc, max_test_auc = 0.0, 0.0
  best_epoch = [0, 0, 0, 0]
  best_val_metric = [0, 0, 0, 0]
  best_test_metric = [0, 0, 0, 0]
  max_test_metric = [0, 0, 0, 0]
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  
  # Epoch Loop
  for epoch in range(args.n_epoch):
    start_epoch = time.time()
    
    # Initialize memory of the model at each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    tgn = tgn.eval()
    decoder = decoder.train()
    loss = 0
    
    # Batch Loop
    for k in range(num_batch):
      s_idx = k * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)

      sources_batch = train_data.sources[s_idx: e_idx]
      destinations_batch = train_data.destinations[s_idx: e_idx]
      timestamps_batch = train_data.timestamps[s_idx: e_idx]
      edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
      labels_batch = train_data.labels[s_idx: e_idx]

      size = len(sources_batch)

      decoder_optimizer.zero_grad()
      with torch.no_grad():
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                     destinations_batch,
                                                                                     destinations_batch,
                                                                                     timestamps_batch,
                                                                                     edge_idxs_batch,
                                                                                     NUM_NEIGHBORS)

      labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
      pred, _, _ = decoder(source_embedding)
      pred = pred.sigmoid()
      decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
      decoder_loss.backward()
      decoder_optimizer.step()
      loss += decoder_loss.item()
    # End Batch Loop
    train_loss_l.append(loss / num_batch)
    
    # 因为要再算一遍train的内容, 所以reset memory
    if USE_MEMORY:
      tgn.memory.__init_memory__()
      
    train_metric, train_emb, train_prob, train_h1, train_h2 = eval_node_classification(
        tgn, decoder, train_data, full_data.edge_idxs, BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    val_metric, val_emb, val_prob, val_h1, val_h2 = eval_node_classification(
        tgn, decoder, val_data, full_data.edge_idxs, BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    test_metric, test_emb, test_prob, test_h1, test_h2 = eval_node_classification(
        tgn, decoder, test_data, full_data.edge_idxs, BATCH_SIZE, n_neighbors=NUM_NEIGHBORS)
    for i in range(len(METRICS)):
      if val_metric[i] > best_val_metric[i]:
        best_val_metric[i] = val_metric[i]
        best_epoch[i] = epoch
        best_test_metric[i] = test_metric[i]
      if test_metric[i] > max_test_metric[i]:
        max_test_metric[i] = test_metric[i]
    val_auc, val_ap, val_f1, val_recall = val_metric
    test_auc, test_ap, test_f1, test_recall = test_metric
    
    ''' save embeds '''
    save_embs(epoch, (train_emb, val_emb, test_emb), (train_prob, val_prob, test_prob), 
              (train_h1, val_h1, test_h1), (train_h2, val_h2, test_h2))
    
    val_auc_l.append(val_auc)
    pickle.dump({
      "val_aps": val_auc_l,
      "train_losses": train_loss_l,
      "epoch_times": [0.0],
      "new_nodes_val_aps": [],
    }, open(results_path, "wb"))

    logger.info(f'Epoch {epoch}: train_loss: {(loss/num_batch):.4f}, val_auc: {val_auc:.4f}, ' + \
                f'test_auc: {test_auc:.4f}, time: {(time.time() - start_epoch):.4f}')

    # if args.use_validation:
    #   if early_stopper.early_stop_check(val_auc):
    #     logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
    #     break
    #   else:
    #     torch.save(decoder.state_dict(), get_checkpoint_path(epoch))
  # End Epoch Loop
  if args.use_validation:
    for i in range(len(METRICS)):
      logger.info(f'{METRICS[i]}: best_epoch={best_epoch[i]}, val_{METRICS[i]}={best_val_metric[i]:.4f}, ' + \
                    f'test_{METRICS[i]}={best_test_metric[i]:.4f}, max_test_{METRICS[i]}={max_test_metric[i]:.4f}')
    best_epoch_l.append(best_epoch)
    best_val_metric_l.append(best_val_metric)
    best_test_metric_l.append(best_test_metric)
    max_test_metric_l.append(max_test_metric)
  else:
    # If we are not using a validation set, the test performance is just the performance computed
    # in the last epoch
    test_auc = val_auc_l[-1]
  pickle.dump({
    "val_aps": val_auc_l,
    "test_ap": test_auc,
    "train_losses": train_loss_l,
    "epoch_times": [0.0],
    "new_nodes_val_aps": [],
    "new_node_test_ap": 0,
  }, open(results_path, "wb"))
  logger.info(f'\n =========================== RUN {i} END ==================================')
  if args.use_validation:
    logger.info(f'test_auc: {test_metric[0]:.4f}, test_ap: {test_metric[1]:.4f}, test_f1: {test_metric[2]:.4f}, test_recall: {test_metric[3]:.4f}')
  else:
    logger.info(f'test_auc: {test_auc:.4f}')
# End Run Loop


if args.use_validation:
  logger.info(f'\n =========================== SUMMARY ==================================')
  for i in range(len(METRICS)):
    logger.info(f'<<< {METRICS[i]} >>>')
    for i_run in range(args.n_runs):
      logger.info(f'RUN #{i_run}: best_epoch={best_epoch_l[i_run][i]}, best_val_{METRICS[i]}={best_val_metric_l[i_run][i]:.4f}, ' + \
                  f'best_test_{best_test_metric_l[i_run][i]:.4f}, max_test_{METRICS[i]}={max_test_metric_l[i_run][i]:.4f}')
    
    logger.info(f'ALL IN ALL -- ave_best_epoch: {round(sum(best_epoch_l[:][i])/args.n_runs, 4)}')
    logger.info(f'ALL IN ALL -- ave_best_val_{METRICS[i]}: {round(sum([x[i] for x in best_val_metric_l])/args.n_runs, 4)}')
    logger.info(f'ALL IN ALL -- ave_best_test_{METRICS[i]}: {round(sum([x[i] for x in best_test_metric_l])/args.n_runs, 4)}' + \
                f' \u00B1 {round(np.std([x[i] for x in best_test_metric_l]), 4)}')
    logger.info(f'ALL IN ALL -- ave_max_test_{METRICS[i]}: {round(sum([x[i] for x in max_test_metric_l])/args.n_runs, 4)}' + \
                f' \u00B1 {round(np.std([x[i] for x in max_test_metric_l]), 4)}')
  
