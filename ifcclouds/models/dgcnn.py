import torch.nn as nn

class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, self.emb_dims, 1),
        )
        self.fc1 = nn.Linear(self.emb_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 13)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.dp3 = nn.Dropout(p=self.dropout)

    def forward(self, x):
        batch_size = x.size(0)
        n_pts = x.size(2)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2), dim=1)
        x = x.view(batch_size, -1, n_pts)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = x3.view