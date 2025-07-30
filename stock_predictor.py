import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42) 

# 1. 数据预处理
class StockDataProcessor:
    def __init__(self, lookback=10):
        self.lookback = lookback  # 用过去10天的数据预测未来1天
        self.scalers = {}  # 存储每只股票的归一化参数
    
    def normalize(self, df, features):
        """对数据进行Min-Max归一化，按股票代码分组处理"""
        normalized_df = df.copy()
        
        for stock_code, group in df.groupby('ts_code'):
            self.scalers[stock_code] = {}
            for feature in features:
                min_val = group[feature].min()
                max_val = group[feature].max()
                if max_val == min_val:
                    normalized = np.zeros_like(group[feature].values)
                else:
                    normalized = (group[feature] - min_val) / (max_val - min_val)
                
                self.scalers[stock_code][feature] = {'min': min_val, 'max': max_val}
                normalized_df.loc[group.index, feature] = normalized
        
        return normalized_df
    
    def create_sequences(self, df, features, target_close_col, target_pct_col):
        """创建        创建序列数据
        输入: 过去lookback天的特征
        输出: 未来1天的收盘价(回归目标)和涨跌幅分类(分类目标)
        """
        sequences = []
        targets_reg = []  # 回归目标：收盘价
        targets_cls = []  # 分类目标：涨跌幅类别
        
        # 按股票代码分组处理
        for stock_code, group in df.groupby('ts_code'):
            group = group.sort_values('trade_date')  # 确保数据按日期排序
            values = group[features].values
            close_values = group[target_close_col].values
            pct_values = group[target_pct_col].values
            
            # 创建序列
            for i in range(len(values) - self.lookback):
                seq = values[i:i+self.lookback]
                # 回归目标：未来1天的收盘价
                target_reg = close_values[i+self.lookback]
                # 分类目标：未来1天的涨跌幅类别
                target_pct = pct_values[i+self.lookback]
                
                # 将涨跌幅分为三类：跌、平、涨
                if target_pct < -0.01:  # 跌幅超0.01%
                    target_cls = 0
                elif target_pct > 0.01:  # 涨幅超0.01%
                    target_cls = 2
                else:  # 平
                    target_cls = 1
                
                sequences.append(seq)
                targets_reg.append(target_reg)
                targets_cls.append(target_cls)
        
        return np.array(sequences), np.array(targets_reg), np.array(targets_cls)

# 2. 自定义数据集
class StockDataset(Dataset):
    def __init__(self, sequences, targets_reg, targets_cls):
        self.sequences = torch.FloatTensor(sequences)
        self.targets_reg = torch.FloatTensor(targets_reg)
        self.targets_cls = torch.LongTensor(targets_cls).long()  # 分类目标需要是长整数
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets_reg[idx], self.targets_cls[idx]

# 3. 定义模型
class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockPredictionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 回归分支 - 预测收盘价
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 分类分支 - 预测涨跌幅类别
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 3类：跌、平、涨
        )
    
    def forward(self, x):
        # LSTM输出
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        
        # 回归输出
        reg_out = self.regression_head(last_out)
        # 分类输出
        cls_out = self.classification_head(last_out)
        
        return reg_out.squeeze(), cls_out

# 4. 训练函数
def train_model(model, train_loader, val_loader, epochs, learning_rate):
    # 损失函数
    criterion_reg = nn.MSELoss()  # 回归损失
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for sequences, targets_reg, targets_cls in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            reg_pred, cls_pred = model(sequences)
            
            # 计算损失
            loss_reg = criterion_reg(reg_pred, targets_reg)
            loss_cls = criterion_cls(cls_pred, targets_cls)
            # 总损失：回归损失和分类损失的加权和
            loss = 0.7 * loss_reg + 0.3 * loss_cls  # 可调整权重
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sequences.size(0)
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 在验证集上评估
        model.eval()
        val_loss = 0.0
        all_reg_preds = []
        all_cls_preds = []
        all_reg_targets = []
        all_cls_targets = []
        
        with torch.no_grad():
            for sequences, targets_reg, targets_cls in val_loader:
                reg_pred, cls_pred = model(sequences)
                
                loss_reg = criterion_reg(reg_pred, targets_reg)
                loss_cls = criterion_cls(cls_pred, targets_cls)
                loss = 0.7 * loss_reg + 0.3 * loss_cls
                
                val_loss += loss.item() * sequences.size(0)
                
                # 保存预测结果和目标值
                all_reg_preds.extend(reg_pred.numpy())
                all_cls_preds.extend(torch.argmax(cls_pred, dim=1).numpy())
                all_reg_targets.extend(targets_reg.numpy())
                all_cls_targets.extend(targets_cls.numpy())
        
        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 计算验证集上的其他指标
        val_rmse = np.sqrt(mean_squared_error(all_reg_targets, all_reg_preds))
        val_acc = accuracy_score(all_cls_targets, all_cls_preds)
        
        # 打印 epoch 信息
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        print(f'Val RMSE: {val_rmse:.6f} | Val Accuracy: {val_acc:.4f}')
        print('-' * 50)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_stock_model.pth')
    
    return model, train_losses, val_losses

# 5. 测试函数
def test_model(model, test_loader):
    model.eval()
    all_reg_preds = []
    all_cls_preds = []
    all_reg_targets = []
    all_cls_targets = []
    
    with torch.no_grad():
        for sequences, targets_reg, targets_cls in test_loader:
            reg_pred, cls_pred = model(sequences)
            
            all_reg_preds.extend(reg_pred.numpy())
            all_cls_preds.extend(torch.argmax(cls_pred, dim=1).numpy())
            all_reg_targets.extend(targets_reg.numpy())
            all_cls_targets.extend(targets_cls.numpy())
    
    # 计算测试集指标
    test_rmse = np.sqrt(mean_squared_error(all_reg_targets, all_reg_preds))
    test_acc = accuracy_score(all_cls_targets, all_cls_preds)
    cls_report = classification_report(all_cls_targets, all_cls_preds, 
                                      target_names=['跌', '平', '涨'])
    
    print('测试集结果:')
    print(f'RMSE (收盘价预测): {test_rmse:.6f}')
    print(f'准确率 (涨跌预测): {test_acc:.4f}')
    print('分类报告:')
    print(cls_report)
    
    return all_reg_preds, all_cls_preds

# 6. 主函数
def main():
    # 读取数据（替换为你的数据路径）
    df = pd.read_csv('stock/ndsd_300750.SZ.csv')
    # 定义特征列
    features = ['open', 'close', 'high', 'low', 'vol', 
                'amount', 'pct_chg', 'change']

    # 初始化数据处理器
    lookback = 10  # 用过去10天数据预测未来1天
    processor = StockDataProcessor(lookback=lookback)

    # 归一化数据
    normalized_df = processor.normalize(df, features)

    # 创建序列数据
    sequences, targets_reg, targets_cls = processor.create_sequences(
        normalized_df, 
        features, 
        target_close_col='close', 
        target_pct_col='pct_chg'
    )

        
    # 划分训练集、验证集和测试集
    # 先划分训练集和临时集
    X_train, X_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = train_test_split(
        sequences, targets_reg, targets_cls, test_size=0.3, shuffle=False  # 时间序列不打乱
    )
    
    # 再将临时集划分为验证集和测试集
    X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = train_test_split(
        X_temp, y_reg_temp, y_cls_temp, test_size=1/3, shuffle=False  # 时间序列不打乱
    )
    
    # 创建数据集和数据加载器
    batch_size = 32
    train_dataset = StockDataset(X_train, y_reg_train, y_cls_train)
    val_dataset = StockDataset(X_val, y_reg_val, y_cls_val)
    test_dataset = StockDataset(X_test, y_reg_test, y_cls_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    input_size = len(features)  # 输入特征数量
    hidden_size = 64  # LSTM隐藏层大小
    num_layers = 2  # LSTM层数
    model = StockPredictionModel(input_size, hidden_size, num_layers)
    
    # 训练模型
    epochs = 50
    learning_rate = 0.001
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs, learning_rate
    )
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # 加载最佳模型并测试
    model.load_state_dict(torch.load('best_stock_model.pth'))
    reg_preds, cls_preds = test_model(model, test_loader)
    
    # 绘制预测与实际收盘价对比图（取前50个样本）
    plt.figure(figsize=(12, 6))
    plt.plot(y_reg_test[:50], label='实际收盘价', color='blue')
    plt.plot(reg_preds[:50], label='预测收盘价', color='red', linestyle='--')
    plt.title('收盘价预测 vs 实际值')
    plt.xlabel('样本索引')
    plt.ylabel('归一化收盘价')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
