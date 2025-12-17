import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal

# ==========================================
# 1. 설정 (Hyperparameters) - High Capacity
# ==========================================
NUM_USERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [설정 유지] ResNet Large + Addition Fusion
D_MODEL = 256       
NUM_HEADS = 8       
BATCH_SIZE = 128    
LR = 0.001          
EPOCHS = 30         

LAMBDA_A = 1.0      
LAMBDA_S = 0.005    

print(f"🚀 학습 시작 | 장치: {DEVICE} | Mode: Large Capacity + Addition + 9:1 Split", flush=True)

# ==========================================
# 2. 데이터셋 클래스 (그대로 유지)
# ==========================================
class MultiModalDataset(Dataset):
    def __init__(self, data_folder, window_size=300, stride=50, fs=128):
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.samples_ppg = []
        self.samples_temp = []
        self.labels = []
        
        search_pattern = os.path.join(data_folder, "user_*.csv")
        file_list = glob.glob(search_pattern)
        
        if not file_list:
            print(f"❌ 경고: '{data_folder}' 경로에서 파일을 하나도 찾지 못했습니다.")
            return

        print(f"📂 데이터 로딩 시작... (총 {len(file_list)}개 파일 감지)", flush=True)
        
        for filepath in file_list:
            filename = os.path.basename(filepath)
            try:
                parts = filename.split('_')
                user_num = int(parts[1])
                label = user_num - 1
            except Exception as e:
                print(f"⚠️ 파일명 파싱 실패 ({filename}): {e}")
                continue

            try:
                df = pd.read_csv(filepath)
                df.columns = [c.strip() for c in df.columns]
                
                if 'PPG' not in df.columns or 'temperature' not in df.columns:
                     continue

                raw_ppg = df['PPG'].values
                raw_temp = df['temperature'].values
                
                processed_ppg = self.preprocess_ppg(raw_ppg)
                processed_temp = (raw_temp - 25.0) / (40.0 - 25.0) 

                num_windows = (len(processed_ppg) - self.window_size) // self.stride
                if num_windows <= 0: continue

                for i in range(num_windows):
                    start = i * self.stride
                    end = start + self.window_size
                    
                    ppg_window = processed_ppg[start:end]
                    if np.max(np.abs(ppg_window)) > 5: continue
                        
                    self.samples_ppg.append(ppg_window)
                    self.samples_temp.append(processed_temp[start:end])
                    self.labels.append(label)
                
            except Exception as e:
                print(f"  ❌ 데이터 로드 에러 ({filename}): {e}")

        self.samples_ppg = np.array(self.samples_ppg, dtype=np.float32)
        self.samples_temp = np.array(self.samples_temp, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        if len(self.labels) > 0:
            self.samples_ppg = np.expand_dims(self.samples_ppg, axis=1)
            self.samples_temp = np.expand_dims(self.samples_temp, axis=1)

        print(f"🎉 전체 데이터 로드 완료! 총 샘플 수: {len(self.labels)}", flush=True)

    def preprocess_ppg(self, signal_data):
        detrended = signal.detrend(signal_data)
        nyquist = 0.5 * self.fs
        cutoff = 4.0 / nyquist
        b, a = signal.butter(4, cutoff, btype='low')
        filtered = signal.filtfilt(b, a, detrended)
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
        return normalized

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples_ppg[idx]), 
                torch.from_numpy(self.samples_temp[idx])), torch.tensor(self.labels[idx])

# ==========================================
# 3. 모델 아키텍처 (ResNet Large)
# ==========================================
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetEncoder_Large(nn.Module):
    def __init__(self, in_channels=1, d_model=256):
        super(ResNetEncoder_Large, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # [Large] 3-4-6 Structure
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, d_model, 6, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_planes, out_planes, blocks, stride):
        layers = []
        layers.append(BasicBlock1D(in_planes, out_planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        seq_out = x.transpose(1, 2)
        pooled_out = self.pool(x).squeeze(-1)
        return pooled_out, seq_out

class LargeFusionModel_Addition(nn.Module):
    def __init__(self, num_users=16, d_model=256, num_heads=8):
        super(LargeFusionModel_Addition, self).__init__()
        self.ppg_encoder = ResNetEncoder_Large(in_channels=1, d_model=d_model)
        self.temp_encoder = ResNetEncoder_Large(in_channels=1, d_model=d_model)
        
        self.cross_att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # [Addition Fusion]
        self.fusion_fc = nn.Linear(d_model, d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(d_model, num_users)
        )

    def forward(self, x_ppg, x_temp):
        z_p, seq_p = self.ppg_encoder(x_ppg)
        z_t, seq_t = self.temp_encoder(x_temp)
        
        query_p = z_p.unsqueeze(1)
        query_t = z_t.unsqueeze(1)
        
        attn_out_p, _ = self.cross_att(query_p, seq_t, seq_t)
        z_p_refined = self.norm_p(z_p + attn_out_p.squeeze(1))
        
        attn_out_t, _ = self.cross_att(query_t, seq_p, seq_p)
        z_t_refined = self.norm_t(z_t + attn_out_t.squeeze(1))
        
        # Addition Fusion
        z_fused = z_p_refined + z_t_refined
        z_fused = F.relu(self.fusion_fc(z_fused))
        
        logits = self.classifier(z_fused)
        return logits, z_p_refined, z_t_refined

# ==========================================
# 4. 손실 함수
# ==========================================
class AlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(AlignmentLoss, self).__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
    def forward(self, z_a, z_b):
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)
        logits = torch.matmul(z_a, z_b.T) / self.temperature
        labels = torch.arange(z_a.size(0)).to(z_a.device)
        return self.ce(logits, labels)

class SpreadControlLoss(nn.Module):
    def __init__(self, threshold=0.001):
        super(SpreadControlLoss, self).__init__()
        self.threshold = threshold
    def forward(self, z):
        z = F.normalize(z, dim=1)
        var = torch.var(z, dim=0).mean()
        return F.relu(var - self.threshold)

# ==========================================
# 5. 실행 (9:1 Split + Validation)
# ==========================================
if __name__ == "__main__":
    data_folder = "./data/PPG_ECG_Data"
    
    # 1. 데이터 로드
    dataset = MultiModalDataset(data_folder, window_size=300, stride=50)
    
    if len(dataset) > 0:
        # [추가됨] 90:10 분할
        train_len = int(0.9 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        
        # DataLoader 분리
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"📊 데이터 분할: Train {len(train_set)} / Val {len(val_set)}", flush=True)

        # 모델 초기화 (Large + Addition)
        model = LargeFusionModel_Addition(num_users=NUM_USERS, d_model=D_MODEL, num_heads=NUM_HEADS).to(DEVICE)
        
        criterion_cls = nn.CrossEntropyLoss()
        criterion_align = AlignmentLoss()
        criterion_spread = SpreadControlLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        print("🚀 학습 시작 (Validation 포함)...", flush=True)
        
        total_batches = len(train_loader)

        for epoch in range(EPOCHS):
            # --- Training ---
            model.train()
            train_loss = 0
            correct = 0
            total_samples = 0
            
            for batch_idx, ((ppg, temp), labels) in enumerate(train_loader):
                ppg, temp, labels = ppg.to(DEVICE), temp.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs, z_p, z_t = model(ppg, temp)
                
                loss_cls = criterion_cls(outputs, labels)
                loss_align = criterion_align(z_p, z_t) + criterion_align(z_t, z_p)
                loss_spread = criterion_spread(z_p) + criterion_spread(z_t)
                
                loss = loss_cls + (LAMBDA_A * loss_align) + (LAMBDA_S * loss_spread)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (batch_idx + 1) % 1000 == 0:
                    cur_loss = train_loss / (batch_idx + 1)
                    cur_acc = 100 * correct / total_samples
                    print(f"   [Train] Batch {batch_idx+1}/{total_batches} | Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}%", flush=True)

            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * correct / total_samples

            # --- Validation ---
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for (ppg, temp), labels in val_loader:
                    ppg, temp, labels = ppg.to(DEVICE), temp.to(DEVICE), labels.to(DEVICE)
                    outputs, _, _ = model(ppg, temp)
                    
                    # Val은 Classification Loss만 주로 확인
                    loss = criterion_cls(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            print(f"✨ [Epoch {epoch+1}/{EPOCHS} DONE]")
            print(f"   Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"   Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%", flush=True)
            print("-" * 60, flush=True)

        print("🏁 학습 완료!", flush=True)
