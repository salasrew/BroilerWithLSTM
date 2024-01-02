import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('washedV2.xlsx')
df['日期'] = pd.to_datetime(df['日期'])
df['日期'] = (df['日期'] - df['日期'].min()).dt.days
df = df[df['日齡'] > 0]

# 日齡	日期	雞隻數	標準毛雞重(g)	毛雞重(g)	標準重量差	增重(g)	均勻度
features = df[['日齡', '日期','標準毛雞重(g)', '毛雞重(g)','標準重量差' , '增重(g)','均勻度' ]]

# scaler = StandardScaler()
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

def create_sequences(data, input_sequence_length, output_sequence_length):
    input_sequences = []
    output_sequences = []
    for i in range(len(data) - input_sequence_length - output_sequence_length + 1):
        input_seq = data[i:(i + input_sequence_length)]
        output_seq = data[i + input_sequence_length:i + input_sequence_length + output_sequence_length]
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)
    return np.array(input_sequences), np.array(output_sequences)

input_sequence_length = 30
output_sequence_length = 1


# 使用 create_sequences 函数转换数据
X, y = create_sequences(features, input_sequence_length, output_sequence_length)
print("Sequence shapes:", X.shape, y.shape)  # 检查序列形状

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 确保TensorDataset可以正确创建
print("Tensor shapes:", X_train.shape, y_train.shape)  # 再次检查形状
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 128  # 或者您希望的其他大小
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    # def forward(self, input_seq):
    #     lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
    #     predictions = self.linear(lstm_out.view(len(input_seq), -1))
    #     return predictions[-1]
    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        hidden_state = torch.zeros(1, batch_size, self.hidden_layer_size)
        cell_state = torch.zeros(1, batch_size, self.hidden_layer_size)
        self.hidden_cell = (hidden_state, cell_state)

        input_seq = input_seq.transpose(0, 1)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        
        # 获取最后一个时间步的输出
        lstm_out_last_step = lstm_out[-1]  # lstm_out的形状为 (sequence_length, batch_size, hidden_size)
        
        predictions = self.linear(lstm_out_last_step)
        return predictions

    
model = LSTMModel(input_size=7)
# model = LSTMModel(input_size=210)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 初始化損失記錄和参数
epoch_losses = []
iterator_losses = []
best_loss = float('inf')
no_improve_count = 0
max_no_improve_epochs = 200

epochs = 150  # 可調整

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(inputs)
        loss = loss_function(y_pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        iterator_losses.append(loss.item())  # 紀錄每次迭代的損失

    epoch_loss = total_loss / len(train_loader)
    epoch_losses.append(epoch_loss)  # 紀錄每個epoch的平均損失

    print(f'Epoch {epoch+1} Average Loss: {epoch_loss:.4f}')

    # 检查损失是否有改善
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        no_improve_count = 0
    else:
        no_improve_count += 1

    # 如果连续一定次数的epoch没有改善，则停止训练
    if no_improve_count >= max_no_improve_epochs:
        print(f"No improvement in {max_no_improve_epochs} epochs, stopping training.")
        break


with torch.no_grad():
    total_test_loss = 0.0
    for inputs, labels in test_loader:
        y_pred = model(inputs)
        loss = loss_function(y_pred, labels)
        total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Test loss: {avg_test_loss:.4f}')

# 繪製每個epoch的平均損失
plt.figure(figsize=(10, 4))
plt.plot(epoch_losses, label='Epoch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch Loss Over Time')
plt.legend()
plt.show()

# 繪製每次迭代的損失
plt.figure(figsize=(10, 4))
plt.plot(iterator_losses, label='Iterator Loss', color='orange')
plt.xlabel('Iterator')
plt.ylabel('Loss')
plt.title('Iterator Loss Over Time')
plt.legend()
plt.show()
