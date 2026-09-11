import torch
from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader


def data_split(dataset, train_perc, val_perc):
    length = len(dataset)
    dataset = dataset.shuffle()
    train_end = int(length * train_perc)
    val_end = train_end + int(length * val_perc)

    train_data = dataset[:train_end]
    val_data = dataset[train_end:val_end]
    test_data = dataset[val_end:]
    train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=32)
    test_loader = DataLoader(test_data,batch_size=32)
    return train_loader, val_loader,test_loader

from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
import torch.nn as nn

def make_mlp(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(out_channels, out_channels),)


class BindingGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GINEConv(make_mlp(9,128),edge_dim=3)
        self.conv2 = GINEConv(make_mlp(128,128),edge_dim=3)
        self.conv3 = GINEConv(make_mlp(128,128),edge_dim=3)
        self.fc = torch.nn.Linear(128,1)

    def forward(self,x,edge_index,edge_attr,batch):
        x = x.float()
        edge_attr = edge_attr.float()
        x = self.conv1(x, edge_index,edge_attr)
        x = F.relu(x)
        x = self.conv2(x,edge_index,edge_attr)
        x = F.relu(x)
        x = self.conv3(x,edge_index,edge_attr)
        x = F.relu(x)
        x = global_add_pool(x,batch)
        x = self.fc(x)
        return x

from sklearn.metrics import roc_auc_score
def evaluate(model,criterion,loader,device):
    model.eval()
    total_loss = 0
    all_probs =[]
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = batch.x.float()
            batch.edge_attr = batch.edge_attr.float()
            logits = model(batch.x, batch.edge_index, batch.edge_attr,batch.batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs>0.5).float()
            correct += (preds == batch.y).sum().item()
            total += batch.y.numel()
            all_probs.append(probs.cpu())
            all_labels.append(batch.y.cpu())


    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_probs)
    accuracy = correct / total
    pr_auc = average_precision_score(all_labels,all_probs)

    return total_loss / len(loader), auc, pr_auc, accuracy


def train_model(model,criterion,optimizer,epochs,dataset,device):
    train_load,val_load,test_load = data_split(dataset,0.8,0.1)
    model.train()
    print('training...')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_load:
            batch = batch.to(device)
            batch.x = batch.x.float()
            batch.edge_attr = batch.edge_attr.float()
            optimizer.zero_grad()
            prediction = model(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
            loss = criterion(prediction,batch.y) #y is label
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_load)
        val_loss,val_auc, pr_auc, acc = evaluate(model,criterion,val_load,device)
        print(f'Epoch {epoch}: train={train_loss},val={val_loss},val_auc={val_auc},val_pr_auc={pr_auc},val_accuracy={acc}')

    test_loss,test_auc,test_pr_auc,test_acc = evaluate(model,criterion,test_load,device)
    print(f'Test loss: {test_loss},Test_AUC: {test_auc},test_pr_auc: {test_pr_auc},Test_accuracy: {test_acc}')

def main():

    dataset = MoleculeNet(root='data',name='BACE')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BindingGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_model(model,criterion,optimizer,200,dataset,device)

if __name__ == "__main__":
    main()
