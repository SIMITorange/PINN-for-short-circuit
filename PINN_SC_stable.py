import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# 1. 数据准备（保持不变）
# file_path = "processed_data.xlsx"
file_path = "../processed_data_only_400_600_new.xlsx"
excel_file = pd.ExcelFile(file_path)
data_frames = []
for sheet in excel_file.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        'temperature': 'Temperature',
        'vds': 'Vds',
        'vgs': 'Vgs',
        'ids': 'Ids'
    })[['Ids', 'Vds', 'Vgs', 'Temperature']]
    data_frames.append(df)

combined_df = pd.concat(data_frames, ignore_index=True)
raw_data = combined_df.to_numpy()

inputs_data = raw_data[:, 1:].copy()
inputs_data[:, 2] += 273.15  # 摄氏温度转开尔文
targets_data = raw_data[:, 0].reshape(-1, 1)

input_scaler = StandardScaler()
target_scaler = StandardScaler()
X_scaled = input_scaler.fit_transform(inputs_data)
y_scaled = target_scaler.fit_transform(targets_data)

input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32)
input_std = torch.tensor(input_scaler.scale_, dtype=torch.float32)
target_mean = torch.tensor(target_scaler.mean_[0], dtype=torch.float32)
target_std = torch.tensor(target_scaler.scale_[0], dtype=torch.float32)

dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32),
                        torch.tensor(y_scaled, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# 2. 定义改进的神经网络模型
class PINN(nn.Module):
    def __init__(self, dropout_rate=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # 可学习参数（添加约束）
        self.param1 = nn.Parameter(torch.tensor(110, dtype=torch.float32))  #稳定版本
        # self.param1 = nn.Parameter(torch.tensor(10, dtype=torch.float32))
        self.param2 = nn.Parameter(torch.tensor(2.9, dtype=torch.float32))
        self.param3 = nn.Parameter(torch.tensor(10.6, dtype=torch.float32))
        self.param4 = nn.Parameter(torch.tensor(0.0011, dtype=torch.float32))
        self.param5 = nn.Parameter(torch.tensor(-0.102, dtype=torch.float32))
        self.param6 = nn.Parameter(torch.tensor(10, dtype=torch.float32))
        self.param7 = nn.Parameter(torch.tensor(-0.1, dtype=torch.float32))
        self.param8 = nn.Parameter(torch.tensor(5, dtype=torch.float32))
        self.param9 = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

    def forward(self, x):
        return self.net(x)

model = PINN()

def physics_model(Vds, Vgs, T, param1, param2, param3, param4, param5, param6, param7, param8, param9):
    if isinstance(Vds, (int, float)):
        Vds = torch.tensor([Vds], dtype=torch.float32)
    elif isinstance(Vds, np.ndarray):
        Vds = torch.tensor(Vds, dtype=torch.float32)

    if isinstance(Vgs, (int, float)):
        Vgs = torch.tensor([Vgs], dtype=torch.float32)
    elif isinstance(Vgs, np.ndarray):
        Vgs = torch.tensor(Vgs, dtype=torch.float32)

    if isinstance(T, (int, float)):
        T = torch.tensor([T], dtype=torch.float32)
    elif isinstance(T, np.ndarray):
        T = torch.tensor(T, dtype=torch.float32)
    NET5_value = param3 * 1 /(param1*(T/300)**-0.01 + param6*(T/300)**param2) #稳定版本
    # NET5_value = param3 * 1/(10 + param6*(T/300)**param2)
    NET3_value = -0.004263*T + 3.422579
    p9 = param5
    NET2_value = -0.005 * Vgs + 0.165
    NET1_value = -0.1717 * Vgs + 3.5755
    term3 = (torch.log(1 + torch.exp(Vgs - NET3_value)))**2 - (torch.log(1 + torch.exp(Vgs - NET3_value - (NET2_value * Vds * ((1 + torch.exp(p9 * Vds))**NET1_value)))))**2
    term1 = NET5_value * (Vgs - NET3_value)
    # term2 = 1 + param4*Vds
    term2 = 1 + 0.0005*Vds #稳定版本
    # term2 = 1 + 0.0005*Vds
    return term2 * term1 * term3

# 4. 训练配置（优化器分组）
optimizer = optim.AdamW([
    {'params': model.net.parameters(), 'lr': 0.001},
    {'params': [model.param1, model.param2, model.param3, model.param4, model.param5,model.param6, model.param7, model.param8, model.param9], 'lr': 0.01}
], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
mse_loss = nn.MSELoss()

# 5. 改进的训练循环（修正动态权重）
num_epochs = 500
grad_clip_value = 1
alpha = 0.1  # 调整alpha以更快响应损失变化
w_data = 0.5
w_physics = 0.5  # 初始更侧重物理损失

losses = []
data_metrics = []
physics_metrics = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    metrics = {'data': 0.0, 'physics': 0.0}

    for X_batch, y_batch in dataloader:
        # 数据前向传播
        pred = model(X_batch)
        loss_data = mse_loss(pred, y_batch)

        # 物理约束计算
        Vds_raw = X_batch[:,0]*input_std[0] + input_mean[0]
        Vgs_raw = X_batch[:,1]*input_std[1] + input_mean[1]
        T_raw = X_batch[:,2]*input_std[2] + input_mean[2]

        physics_pred = physics_model(Vds_raw, Vgs_raw, T_raw, model.param1, model.param2, model.param3, model.param4, model.param5, model.param6, model.param7, model.param8, model.param9)
        physics_pred_scaled = (physics_pred - target_mean) / target_std
        loss_physics = mse_loss(pred, physics_pred_scaled.unsqueeze(1))

        # 动态权重调整（修正后的计算方式）
        with torch.no_grad():
            current_data = loss_data.item()
            current_physics = loss_physics.item()
            total = current_data + current_physics + 1e-8
            new_wd = current_data / total  # data权重由数据损失占比决定
            new_wp = current_physics / total  # physics权重由物理损失占比决定
            w_data = alpha * w_data + (1 - alpha) * new_wd
            # w_data = w_data + 0.5
            w_physics = alpha * w_physics + (1 - alpha) * new_wp

        # 总损失
        total_loss_batch = w_data * loss_data + w_physics * loss_physics

        # 反向传播与优化
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()

        # 记录指标
        total_loss += total_loss_batch.item()
        metrics['data'] += current_data
        metrics['physics'] += current_physics

    # 更新学习率
    scheduler.step(total_loss/len(dataloader))

    # 打印训练信息
    avg_loss = total_loss / len(dataloader)
    avg_data_metric = metrics['data'] / len(dataloader)
    avg_physics_metric = metrics['physics'] / len(dataloader)
    losses.append(avg_loss)
    data_metrics.append(avg_data_metric)
    physics_metrics.append(avg_physics_metric)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:04d} | Loss: {avg_loss:.3e} | "
              f"Data: {metrics['data']/len(dataloader):.2e} | "
              f"Physics: {metrics['physics']/len(dataloader):.2e} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"params: {model.param1.item():.3f}, {model.param2.item():.3f}, "
              f"{model.param3.item():.3f}, {model.param4.item():.3f}, {model.param5.item():.3f}, {model.param6.item():.3f}, {model.param7.item():.3f}")
             # 动态绘制损失变化图

# 6. 模型预测与绘图（保持不变）
def neural(Vds, Vgs, T):
    input_arr = np.array([[Vds, Vgs, T + 273.15]])
    scaled_input = input_scaler.transform(input_arr)
    tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
    with torch.no_grad():
        scaled_output = model(tensor_input)
    return target_scaler.inverse_transform(scaled_output.numpy())[0][0]

def plot_results(true_values, predicted_values, physic_values):
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", marker='o', markersize=3)
    plt.plot(predicted_values, label="Predicted Values", marker='x', markersize=3)
    plt.plot(physic_values, label="Physics Values", marker='.', markersize=3)
    plt.xlabel("Sample Index")
    plt.ylabel("Ids")
    plt.title("Comparison of True and Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

true_values = targets_data.flatten()[:900]
predicted_values = [neural(Vds, Vgs, T-273.15) for Vds, Vgs, T in inputs_data[:900]]
physic_values = [physics_model(Vds, Vgs, T, model.param1, model.param2, model.param3, model.param4, model.param5, model.param6, model.param7, model.param8, model.param9)
                 for Vds, Vgs, T in inputs_data[:900]]
physic_values = [tensor.detach().numpy() for tensor in physic_values]

# 训练完成后，绘制loss-epoch图
epochs = list(range(1, num_epochs + 1))  # 生成epoch列表
plt.plot(epochs, np.log10(losses), label='Loss')  # 绘制损失随epoch变化的曲线
plt.plot(epochs, np.log10(data_metrics), label='Data Metric')
plt.plot(epochs, np.log10(physics_metrics), label='Physics Metric')
plt.xlabel('Epoch')  # 设置x轴标签
plt.ylabel('Loss')  # 设置y轴标签
plt.title('Loss vs Epoch')  # 设置图形标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格线
plt.show()  # 显示图形

plot_results(true_values, predicted_values, physic_values)

print(f"params: {model.param1.item():.3f}, {model.param2.item():.3f}, "
      f"{model.param3.item():.3f}, {model.param4.item():.3f}, {model.param5.item():.3f}, {model.param6.item():.3f}, {model.param7.item():.3f}, {model.param8.item():.3f}, {model.param9.item():.3f}")
