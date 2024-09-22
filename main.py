import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

torch.manual_seed(42)

df = pd.read_csv('flu_data.csv')

x_value = df[['Temperature', 'Humidity', 'Precipitation', 'Vaccination Rate', 
               'Population Density', 'Social Distancing Measures', 
               'Previous Flu Cases', 'Age Distribution 0-14', 
               'Age Distribution 15-64', 'Age Distribution 65+']].values
y_value = df['Flu Cases'].values.reshape(-1, 1)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_value = torch.tensor(scaler_x.fit_transform(x_value), dtype=torch.float32)
y_value = torch.tensor(scaler_y.fit_transform(y_value), dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x_value, y_value, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return Data(x=self.x[idx].view(1, -1), edge_index=torch.tensor([[0], [0]], dtype=torch.long), y=self.y[idx])

train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=lambda batch: Batch.from_data_list(batch))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: Batch.from_data_list(batch))

class AdvancedGCRNN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_size, rnn_layers):
        super(AdvancedGCRNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 32)
        self.gat = GATConv(32, 32)
        self.rnn = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=rnn_layers,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.gat(x, edge_index))
        x = x.unsqueeze(1)
        rnn_out, _ = self.rnn(x)
        rnn_out = self.dropout(rnn_out)
        rnn_out = self.batch_norm(rnn_out[:, -1, :])
        return self.fc(rnn_out)

PARAMETERS = {
    "num_node_features": x_value.shape[1],
    "num_classes": 1,
    "hidden_size": 128,
    "rnn_layers": 3,
    "epochs": 200,
}

graph_model = AdvancedGCRNN(PARAMETERS["num_node_features"], PARAMETERS["num_classes"], PARAMETERS["hidden_size"], PARAMETERS["rnn_layers"])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(graph_model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

for epoch in range(PARAMETERS["epochs"]):
    graph_model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = graph_model(data)
        if data.y.dim() == 1:
            data.y = data.y.view(-1, PARAMETERS["num_classes"])
        loss = loss_fn(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f'Epoch [{epoch + 1}/{PARAMETERS["epochs"]}], Loss: {avg_loss:.4f}')

graph_model.eval()
with torch.inference_mode():
    total_test_loss = 0
    for data in test_loader:
        output = graph_model(data)
        if data.y.dim() == 1:
            data.y = data.y.view(-1, PARAMETERS["num_classes"])
        loss = loss_fn(output, data.y)
        total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Validation/Test Loss: {avg_test_loss:.4f}')

predictions = []
actuals = []

with torch.inference_mode():
    for data in test_loader:
        output = graph_model(data)
        predictions.append(output)
        actuals.append(data.y)

predictions = torch.cat(predictions, dim=0).numpy()
actuals = torch.cat(actuals, dim=0).numpy().reshape(-1, 1)

predictions = scaler_y.inverse_transform(predictions)
actuals = scaler_y.inverse_transform(actuals)

mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual Values', linestyle='-')
plt.plot(predictions, label='Predicted Values', linestyle='--')
plt.legend()
plt.title('Actual vs Predicted Flu Cases')
plt.xlabel('Sample Index')
plt.ylabel('Flu Cases')
plt.show()
