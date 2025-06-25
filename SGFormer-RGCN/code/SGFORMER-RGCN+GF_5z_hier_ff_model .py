from torch_geometric.loader import DataLoader
from dataset_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv,RGCNConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool import global_add_pool, global_max_pool, SAGPooling
import os
from sklearn.model_selection import KFold


target = ['lut', 'ff', 'dsp', 'bram', 'uram', 'srl', 'cp', 'power']
tar_idx = 1
jknFlag = 0


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def add_edge_type_to_dataset(dataset_list):
    updated_dataset_list = []

    for data in dataset_list:

        if 'edge_attr' in data:
            edge_attr = data.edge_attr
            edge_type = torch.zeros(edge_attr.size(0), dtype=torch.long)
            edge_type[edge_attr[:, 0] == 0.0] = 0
            edge_type[edge_attr[:, 0] == 1.0] = 1
            edge_type[edge_attr[:, 0] == 4.0] = 2

            data.edge_type = edge_type

        updated_dataset_list.append(data)

    return updated_dataset_list

def full_attention_conv(qs, ks, vs, output_attn=False):
    # qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    # ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    if ks.dim() == 2:
        ks = ks.unsqueeze(1)
    if vs.dim() == 2:
        vs = vs.unsqueeze(1)
    if qs.dim() == 2:
        qs = qs.unsqueeze(1)
    N = qs.shape[0]
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape)
    )  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(
            dim=-1, keepdims=True
        )  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, num_relations=3, use_weight=True):
        super(TransConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.use_weight = use_weight

        self.Wk = torch.nn.ModuleList([Linear(in_channels, out_channels * num_heads, bias=False) for _ in range(num_relations)])
        self.Wq = torch.nn.ModuleList([Linear(in_channels, out_channels * num_heads, bias=False) for _ in range(num_relations)])
        if use_weight:
            self.Wv = torch.nn.ModuleList([Linear(in_channels, out_channels * num_heads, bias=False) for _ in range(num_relations)])

        self.head_proj = Linear(out_channels * num_heads, out_channels, bias=False)

    def forward(self, x, edge_index, edge_type):
        row, col = edge_index
        x_out = torch.zeros_like(x)

        for rel_type in range(self.num_relations):

            mask = (edge_type == rel_type)
            row_rel, col_rel = row[mask], col[mask]


            query = self.Wq[rel_type](x[row_rel]).reshape(-1, self.num_heads, self.out_channels)
            key = self.Wk[rel_type](x[col_rel]).reshape(-1, self.num_heads, self.out_channels)
            if self.use_weight:
                value = self.Wv[rel_type](x[col_rel]).reshape(-1, self.num_heads, self.out_channels)
            else:
                value = x[col_rel].reshape(-1, 1, self.out_channels)


            attention_output = full_attention_conv(query, key, value)
            attn_output = attention_output.mean(dim=1)
            x_out[row_rel] += attn_output
        x_out = self.head_proj(x_out)
        return x_out

class TransConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, num_heads=1, num_relations=3,dropout=0.0, use_bn=True, use_residual=True):
        super(TransConv, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.fcs.append(Linear(in_channels, out_channels))


        for _ in range(num_layers):
            self.convs.append(TransConvLayer(out_channels, out_channels, num_heads, num_relations=num_relations))
            if use_bn:
                self.bns.append(torch.nn.LayerNorm(out_channels))


        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self,x,edge_index,edge_type):
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_ = [x]

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x, edge_index):
        attentions = []
        layer_outputs = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        if self.activation:
            x = self.activation(x)

        layer_outputs.append(x)

        for i, conv in enumerate(self.convs):
            x, attn = conv(x, edge_index, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_outputs[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_outputs.append(x)

        return torch.stack(attentions, dim=0)


class HierNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers,num_relations, use_graph,hls_dim, drop_out=0.0, pool_ratio=0.5):
        super(HierNet, self).__init__()

        self.drop_out = drop_out
        self.pool_ratio = pool_ratio
        self.use_graph = use_graph
        self.graph_weight = torch.nn.Parameter(torch.tensor(0.6))


        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    RGCNConv(in_channels=in_channels, out_channels=out_channels, num_relations=num_relations))
            elif i == 1:
                self.convs.append(RGCNConv(in_channels=out_channels, out_channels=out_channels,
                                           num_relations=num_relations))
            elif i == 2:
                self.convs.append(RGCNConv(in_channels=out_channels, out_channels=out_channels,
                                           num_relations=num_relations))


        self.global_pool = global_add_pool
        self.trans_conv = TransConv(in_channels, out_channels, num_layers=1, num_relations=num_relations,dropout=drop_out)
        self.channels = [out_channels * 1 + hls_dim, 64, 64, 1]
        self.mlps = torch.nn.ModuleList()

        for i in range(len(self.channels) - 1):
            fc = Linear(self.channels[i], self.channels[i + 1])
            self.mlps.append(fc)

    def forward(self, x, edge_index,edge_type,  batch, hls_attr):

        x = x.to(torch.float32)
        trans_x = self.trans_conv(x, edge_index, edge_type)

        for step in range(len(self.convs)):
            x = self.convs[step](x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)

        if self.use_graph:
            x = self.graph_weight * x + (1 - self.graph_weight) * trans_x
        else:
            x = trans_x
        h = self.global_pool(x, batch)

        x = torch.cat([h, hls_attr], dim=-1)

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
    total_mape = 0
    for _, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        hls_attr = data['hls_attr']
        edge_type = data['edge_type']
        out = model(data.x, data.edge_index, edge_type, data.batch, hls_attr)
        out = out.view(-1)
        true_y = data['y'].t()
        mse = F.huber_loss(out, true_y[tar_idx]).float()
        mape = mape_loss(out, true_y[tar_idx]).float()
        loss = mse
        loss.backward()
        optimizer.step()
        total_mse += mse.item() * data.num_graphs
        total_mape += mape.item() * data.num_graphs
    ds = train_loader.dataset
    total_mse = total_mse / len(ds)
    total_mape = total_mape / len(ds)

    return total_mse, total_mape


def test(model, loader, epoch):
    model.eval()
    with torch.no_grad():
        mse = 0
        mape = 0
        y = []
        y_hat = []
        residual = []
        for _, data in enumerate(loader):
            data = data.to(device)
            hls_attr = data['hls_attr']
            edge_type = data['edge_type']
            out = model(data.x, data.edge_index, edge_type, data.batch, hls_attr)
            out = out.view(-1)
            true_y = data['y'].t()
            mse += F.huber_loss(out, true_y[tar_idx]).float().item() * data.num_graphs  # MSE
            mape += mape_loss(out, true_y[tar_idx]).float().item() * data.num_graphs  # MAPE
            y.extend(true_y[tar_idx].cpu().numpy().tolist())
            y_hat.extend(out.cpu().detach().numpy().tolist())
            residual.extend((true_y[tar_idx] - out).cpu().detach().numpy().tolist())
        if epoch % 10 == 0:
            print('pred.y:', out)
            print('data.y:', true_y[tar_idx])
        ds = loader.dataset
        mse = mse / len(ds)
        mape = mape / len(ds)
    return mse, mape


if __name__ == "__main__":
    set_seed(128)
    batch_size = 32
    dataset_dir = os.path.abspath('../dataset/std')
    model_dir = os.path.abspath('./model')
    dataset = os.listdir(dataset_dir)
    dataset_list = generate_dataset(dataset_dir, dataset, print_info=False)
    updated_dataset_list = add_edge_type_to_dataset(dataset_list)

    kfold = KFold(n_splits=5, shuffle=True, random_state=128)

    all_min_train_mape = []
    all_min_test_mape = []

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
        model = HierNet(in_channels=data_ini.num_features, use_graph=True, out_channels=128, num_layers=3,
                        hls_dim=6, drop_out=0.0, num_relations=3)
        model = model.to(device)
        print(model)

        LR = 0.005
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)

        min_train_mape = float('inf')
        min_test_mape = float('inf')

        for epoch in range(500):
            train_loss, train_mape = train(model, train_loader)
            test_loss, test_mape = test(model, test_loader, epoch)
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            print(f'Epoch: {epoch:03d}, Train MAPE: {train_mape:.4f}, Test MAPE: {test_mape:.4f}')

            if epoch % 10 == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.9

            if train_mape < min_train_mape:
                min_train_mape = train_mape

                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'min_train_mape': min_train_mape
                }, os.path.join(model_dir, f'{target[tar_idx]}_fold{fold + 1}_sgformer-rgcn+gf_5z_mae_h64_checkpoint_train.pt'))

            if test_mape < min_test_mape:
                min_test_mape = test_mape

                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'min_test_mape': min_test_mape
                }, os.path.join(model_dir, f'{target[tar_idx]}_fold{fold + 1}_sgformer-rgcn+gf_5z_mae_h64_checkpoint_test.pt'))

        print(f"Fold {fold + 1} Min Train MAPE: {min_train_mape}")
        print(f"Fold {fold + 1} Min Test MAPE: {min_test_mape}")

        all_min_train_mape.append(min_train_mape)
        all_min_test_mape.append(min_test_mape)


    print("\nSummary of Cross-Validation:")
    for i in range(kfold.get_n_splits()):
        print(f"Fold {i + 1} - Min Train MAPE: {all_min_train_mape[i]:.4f}, Min Test MAPE: {all_min_test_mape[i]:.4f}")


    avg_min_train_mape = sum(all_min_train_mape) / len(all_min_train_mape)
    avg_min_test_mape = sum(all_min_test_mape) / len(all_min_test_mape)

    print(f"\nAverage Min Train MAPE: {avg_min_train_mape:.4f}")
    print(f"Average Min Test MAPE: {avg_min_test_mape:.4f}")







