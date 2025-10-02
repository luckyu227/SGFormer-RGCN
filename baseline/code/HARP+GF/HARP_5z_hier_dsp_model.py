from torch_geometric.loader import DataLoader
from dataset_utils import *
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.pool import global_add_pool, global_max_pool, SAGPooling
from sklearn.model_selection import KFold  # 导入KFold
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import (
    TransformerConv,
    JumpingKnowledge,
    SAGPooling,
    global_add_pool,
    global_max_pool,
)
from nn_att import MyGlobalAttention
target = ['lut', 'ff', 'dsp', 'bram', 'uram', 'srl', 'cp', 'power']
tar_idx = 2
jknFlag = 0

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 当前GPU
    torch.cuda.manual_seed_all(seed) #所有GPU

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class HierNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, conv_type, hls_dim,drop_out=0.0, pool_ratio=0.5, jknFlag=True, encode_edge=False, edge_dim=0,
                 separate_P=True, separate_T=False, separate_pseudo=False, separate_icmp=False):
        super(HierNet, self).__init__()
        self.drop_out = drop_out
        self.pool_ratio = pool_ratio
        self.encode_edge = encode_edge
        self.edge_dim = edge_dim
        if conv_type == 'gcn':
            conv = GCNConv
        elif conv_type == 'gat':
            conv = GATConv
        elif conv_type == 'sage':
            conv = SAGEConv
        elif conv_type == 'transformer':
            conv = TransformerConv
        else:
            conv = GCNConv

        self.convs = torch.nn.ModuleList()
        self.num_conv_layers = num_layers
        self.pools = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    TransformerConv(in_channels, hidden_channels,
                                    edge_dim=edge_dim if encode_edge else None,
                                    dropout=drop_out)
                )
            else:
                self.convs.append(
                    TransformerConv(hidden_channels, hidden_channels,
                                    edge_dim=edge_dim if encode_edge else None,
                                    dropout=drop_out)
                )

        self.jknFlag = jknFlag

        if jknFlag:
            self.jkn = JumpingKnowledge('lstm', channels=hidden_channels, num_layers=2)
        self.separate_P = separate_P
        self.separate_T = separate_T
        self.separate_pseudo = separate_pseudo
        self.separate_icmp = separate_icmp

        if self.separate_P:
            self.gate_nn_P = self.node_att_gate_nn(hidden_channels)
            self.glob_P = MyGlobalAttention(self.gate_nn_P, None)

        if self.separate_T:
            self.gate_nn_T = self.node_att_gate_nn(hidden_channels)
            self.glob_T = MyGlobalAttention(self.gate_nn_T, None)

        if self.separate_pseudo:
            self.gate_nn_pseudo = self.node_att_gate_nn(hidden_channels)
            self.glob_pseudo = MyGlobalAttention(self.gate_nn_pseudo, None)

        if self.separate_icmp:
            self.gate_nn_icmp = self.node_att_gate_nn(hidden_channels)
            self.glob_icmp = MyGlobalAttention(self.gate_nn_icmp, None)

        self.global_pool = global_add_pool
        self.channels = [hidden_channels * (separate_P + separate_T + separate_pseudo + separate_icmp)+ hls_dim, 64, 64, 1]
        self.mlps = torch.nn.ModuleList()

        for i in range(len(self.channels) - 1):
            fc = Linear(self.channels[i], self.channels[i + 1])
            self.mlps.append(fc)

    def node_att_gate_nn(self, D):
        return Sequential(Linear(D, D), ReLU(), Linear(D, 1))

    def forward(self, x, edge_index, batch, hls_attr,edge_attr=None):

        x = x.to(torch.float32)
        h_list = []

        for i in range(self.num_conv_layers):
            if self.encode_edge and isinstance(self.convs[i], TransformerConv):
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)
            h_list.append(x)

        if self.jknFlag:
            x = self.jkn(h_list)
        else:
            x = h_list[-1]

        out_g = None
        if self.separate_P:
            out_P, _ = self.glob_P(x, batch)
            out_g = out_P if out_g is None else torch.cat([out_g, out_P], dim=1)

        if self.separate_T:
            out_T, _ = self.glob_T(x, batch)
            out_g = out_T if out_g is None else torch.cat([out_g, out_T], dim=1)

        if self.separate_pseudo:
            out_pseudo, _ = self.glob_pseudo(x, batch)
            out_g = out_pseudo if out_g is None else torch.cat([out_g, out_pseudo], dim=1)

        if self.separate_icmp:
            out_icmp, _ = self.glob_icmp(x, batch)
            out_g = out_icmp if out_g is None else torch.cat([out_g, out_icmp], dim=1)

        if out_g is None:
            out_g = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)

        x=out_g
        x = torch.cat([out_g, hls_attr], dim=-1)


        for f in range(len(self.mlps)):
            if f < len(self.mlps) - 1:
                x = F.relu(self.mlps[f](x))
                x = F.dropout(x, p=self.drop_out, training=self.training)
            else:
                x = self.mlps[f](x)

        return x


def train(model, train_loader):
    model.train()
    total_mse = 0
    total_mae = 0
    for _, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        hls_attr = data['hls_attr']
        out = model(data.x, data.edge_index, data.batch, hls_attr)
        out = out.view(-1)
        true_y = data['y'].t()
        mse = F.huber_loss(out, true_y[tar_idx]).float()
        mae = F.l1_loss(out, true_y[tar_idx]).float()
        loss = mse
        loss.backward()
        optimizer.step()
        total_mse += mse.item() * data.num_graphs
        total_mae += mae.item() * data.num_graphs
    ds = train_loader.dataset
    total_mse = total_mse / len(ds)
    total_mae = total_mae / len(ds)

    return total_mse, total_mae


def test(model, loader, epoch):
    model.eval()
    with torch.no_grad():
        mse = 0
        mae = 0
        y = []
        y_hat = []
        residual = []
        for _, data in enumerate(loader):
            data = data.to(device)
            hls_attr = data['hls_attr']
            out = model(data.x, data.edge_index, data.batch, hls_attr)
            out = out.view(-1)
            true_y = data['y'].t()
            mse += F.huber_loss(out, true_y[tar_idx]).float().item() * data.num_graphs  # MSE
            mae += F.l1_loss(out, true_y[tar_idx]).float().item() * data.num_graphs  # MAE
            y.extend(true_y[tar_idx].cpu().numpy().tolist())
            y_hat.extend(out.cpu().detach().numpy().tolist())
            residual.extend((true_y[tar_idx] - out).cpu().detach().numpy().tolist())
        if epoch % 10 == 0:
            print('pred.y:', out)
            print('data.y:', true_y[tar_idx])
        ds = loader.dataset
        mse = mse / len(ds)
        mae = mae / len(ds)
    return mse, mae


if __name__ == "__main__":
    set_seed(128)
    batch_size = 32
    dataset_dir = os.path.abspath('../dataset/std')
    model_dir = os.path.abspath('./model')
    dataset = os.listdir(dataset_dir)
    dataset_list = generate_dataset(dataset_dir, dataset, print_info=True)

    print(f"Dataset loaded with {len(dataset_list)} graphs")

    kfold = KFold(n_splits=5, shuffle=True, random_state=128)  # 使用5折交叉验证

    all_min_train_mae = []
    all_min_test_mae = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset_list)):
        print(f"Fold {fold + 1}")
        train_ds = [dataset_list[i] for i in train_idx]
        test_ds = [dataset_list[i] for i in test_idx]

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        data_ini = None
        for step, data in enumerate(train_loader):
            if step == 0:
                data_ini = data
                break

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HierNet(in_channels=data_ini.num_features, hidden_channels=64, num_layers=3, conv_type='transformer',
                        hls_dim=6,drop_out=0.0)
        model = model.to(device)
        print(model)

        LR = 0.005
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)
        tolerance = 0.05


        min_train_mae = float('inf')
        min_test_mae = float('inf')

        for epoch in range(500):
            train_loss, train_mae = train(model, train_loader)
            test_loss, test_mae = test(model, test_loader, epoch)
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            print(f'Epoch: {epoch:03d}, Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}')

            if epoch % 10 == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.9
            if (train_mae < test_mae) or (abs(train_mae - test_mae) <= tolerance):
                if train_mae < min_train_mae:
                    min_train_mae = train_mae
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'min_train_mae': min_train_mae
                    }, os.path.join(model_dir, f'{target[tar_idx]}_fold{fold + 1}_harp+gf_5z_mae_h64_checkpoint_train.pt'))
                if test_mae < min_test_mae:
                    min_test_mae = test_mae
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'min_test_mae': min_test_mae
                    }, os.path.join(model_dir, f'{target[tar_idx]}_fold{fold + 1}_harp+gf_5z_mae_h64_checkpoint_test.pt'))

        print(f"Fold {fold + 1} Min Train MAE: {min_train_mae}")
        print(f"Fold {fold + 1} Min Test MAE: {min_test_mae}")

        all_min_train_mae.append(min_train_mae)
        all_min_test_mae.append(min_test_mae)

    print("\nSummary of Cross-Validation:")
    for i in range(kfold.get_n_splits()):
        print(f"Fold {i + 1} - Min Train MAE: {all_min_train_mae[i]:.4f}, Min Test MAE: {all_min_test_mae[i]:.4f}")

    avg_min_train_mae = sum(all_min_train_mae) / len(all_min_train_mae)
    avg_min_test_mae = sum(all_min_test_mae) / len(all_min_test_mae)

    print(f"\nAverage Min Train MAE: {avg_min_train_mae:.4f}")
    print(f"Average Min Test MAE: {avg_min_test_mae:.4f}")

