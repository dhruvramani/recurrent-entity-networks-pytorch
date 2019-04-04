import os
import gc
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from models import *
from dataset import CommonSenseDataset
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CommonSense Base Model')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate') # NOTE change for diff models
parser.add_argument('--batch_size', default=30, type=int)
parser.add_argument('--resume', '-r', type=int, default=1, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--momentum', '-lm', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-ld', type=float, default=1e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--model', default='birnn', help='Model : birnn/rowcnn')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch, step = 0, 0
best_p, best_r, best_f = 0.0, 0.0, 0.0
loss_fn = torch.nn.BCELoss()
dataset = CommonSenseDataset(10)

print('==> Creating network..')
if(args.model == 'birnn'):
    net = BiRNN() # TODO Change here
else :
    net = RowCNN()
net = net.to(device)

if(args.resume):
    if(os.path.isfile('../save/{}.ckpt'.format(args.model))):
        net.load_state_dict(torch.load('../save/{}.ckpt'.format(args.model)))
        print('==> CommonSenseNet : loaded')

    if(os.path.isfile("../save/{}_info.txt".format(args.model))):
        with open("../save/{}_info.txt".format(args.model), "r") as f:
            epoch, step = (int(i) for i in str(f.read()).split(" "))
        print("=> CommonSenseNet : prev epoch found")
else :
    with open("../save/logs/{}_train_loss.log".format(args.model), "w+") as f:
        pass 

def train(epoch):
    global step
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)

    train_loss, total, prec, recall, fscore = 0, 0, 0.0, 0.0, 0.0
    params = list(net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.decay)
    
    for i in range(step, len(dataloader)):
        sequences, predictions = next(dataloader)
        if(args.model == 'rowcnn'):
            sequences = sequences.unsqueeze(1) # NOTE : Comment for BiRNN
        sequences, predictions = sequences.type(torch.FloatTensor).to(device), predictions.type(torch.FloatTensor).to(device)
        output = net(sequences)
        optimizer.zero_grad()
        loss = loss_fn(output, predictions) # Last LSTM output, Prediction
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()

        pred_num, out_num = predictions.cpu().detach().numpy(), output.cpu().detach().numpy()
        out_num[out_num > 0.5] = 1.
        out_num[out_num < 0.5] = 0.

        prec += precision_score(pred_num, out_num, average='macro')
        recall += recall_score(pred_num, out_num, average='macro')
        fscore += f1_score(pred_num, out_num, average='macro')

        with open("../save/logs/{}_train_loss.log".format(args.model), "a+") as lfile:
            lfile.write("{}\n".format(train_loss / (i - step +1)))

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(net.state_dict(), '../save/{}.ckpt'.format(args.model))

        with open("../save/{}_info.txt".format(args.model), "w+") as f:
            f.write("{} {}".format(epoch, i))

        progress_bar(i, len(dataloader), 'Loss: %.3f' % (train_loss / (i - step + 1)))

    step = 0
    print('=> Training : Epoch [{}/{}], Loss:{:.4f}\nPrecision : {}, Recall : {}, F1 Score : {}'.format(epoch + 1, args.epochs, train_loss / len(dataloader), prec / len(dataloader), recall / len(dataloader), fscore / len(dataloader)))

def test(epoch):
    global best_p, best_f, best_r
    dataset = CommonSenseDataset(10, tr="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)

    prec, recall, fscore = 0.0, 0.0, 0.0  

    for i in range(step, len(dataloader)):
        sequences, predictions = next(dataloader)
        if(args.model == 'rowcnn'):
            sequences = sequences.unsqueeze(1) # NOTE : Comment for BiRNN
        sequences, predictions = sequences.type(torch.FloatTensor).to(device), predictions.type(torch.FloatTensor).to(device)
        output = net(sequences)

        pred_num, out_num = predictions.cpu().detach().numpy(), output.cpu().detach().numpy()
        out_num[out_num > 0.5] = 1.
        out_num[out_num < 0.5] = 0.

        prec += precision_score(pred_num, out_num, average='macro')
        recall += recall_score(pred_num, out_num, average='macro')
        fscore += f1_score(pred_num, out_num, average='macro')

    if(best_p < prec / len(dataloader)):
        best_p = prec / len(dataloader)
    if(best_r < recall / len(dataloader)):
        best_r = recall / len(dataloader)
    if(best_f < fscore / len(dataloader)):
        best_f = fscore / len(dataloader)

    print('=> Test : Epoch [{}/{}], Precision : {}, Recall : {}, F1 Score : {}'.format(epoch + 1, args.epochs, prec / len(dataloader), recall / len(dataloader), fscore / len(dataloader)))


for epoch in range(epoch, epoch + args.epochs):
    train(epoch)
    test(epoch)
    print('=> Best - Precision : {}, Recall : {}, F1 Score : {}'.format(best_p, best_r, best_f))
