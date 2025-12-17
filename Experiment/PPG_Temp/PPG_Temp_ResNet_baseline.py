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
from tqdm.auto import tqdm

# ==========================================
# 1. 설정 (Hyperparameters)
# ==========================================
NUM_USERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [모델 설정] 사용자가 요청한 Baseline ResNet + Addition 설정
D_MODEL = 128       # ResNet 기본 채널
NUM_HEADS = 4       
BATCH_SIZE = 128    # 배치 사이즈 (VRAM에 맞춰 조절)
LR = 0.001          
EPOCHS = 40         

# [손실 가중치]
LAMBDA_A = 0.5      
LAMBDA_S = 0.01     

print(f"🚀 학습 시작 | 장치: {DEVICE} | Mode: ResNet Baseline + Addition + Validation", flush=True)

# ==========================================
# 2. 데이터셋 클래스
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
            print(f"❌ 데이터 없음: {data_folder}")
            return

        print(f"📂 데이터 로딩 중... ({len(file_list)} files)")
        
        for filepath in file_list:
            try:
                filename = os.path.basename(filepath)
                parts = filename.split('_')
                user_num = int(parts[1])
                label = user_num - 1
                
                df = pd.read_csv(filepath)
                df.columns = [c.strip() for c in df.columns]
                
                if 'PPG' not in df.columns or 'temperature' not in df.columns:
                    continue

                raw_ppg = df['PPG'].values
                raw_temp = df['temperature'].values
                
                # 전처리
                detrended = signal.detrend(raw_ppg)
                b, a = signal.butter(4, 4.0/(0.5*fs), btype='low')
                filtered = signal.filtfilt(b, a, detrended)
                processed_ppg = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
                
                processed_temp = (raw_temp - 25.0) / (40.0 - 25.0) 

                num_windows = (len(processed_ppg) - window_size) // stride
                if num_windows <= 0: continue

                for i in range(num_windows):
                    start = i * stride
                    end = start + window_size
                    ppg_win = processed_ppg[start:end]
                    if np.max(np.abs(ppg_win)) > 5: continue
                        
                    self.samples_ppg.append(ppg_win)
                    self.samples_temp.append(processed_temp[start:end])
                    self.labels.append(label)
                    
            except Exception as e:
                print(f"❌ 로드 에러 {filename}: {e}")

        self.samples_ppg = np.array(self.samples_ppg, dtype=np.float32)
        self.samples_temp = np.array(self.samples_temp, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        if len(self.labels) > 0:
            self.samples_ppg = np.expand_dims(self.samples_ppg, axis=1)
            self.samples_temp = np.expand_dims(self.samples_temp, axis=1)

        print(f"🎉 로드 완료! 총 {len(self.labels)} 샘플")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples_ppg[idx]), 
                torch.from_numpy(self.samples_temp[idx])), torch.tensor(self.labels[idx])

# ==========================================
# 3. 모델 아키텍처 (ResNet Baseline + Addition)
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
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=1, d_model=128):
        super(ResNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Baseline 구조 [2, 2, 2]
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, d_model, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_planes, out_planes, blocks, stride):
        layers = [BasicBlock1D(in_planes, out_planes, stride)]
        for _ in range(1, blocks): layers.append(BasicBlock1D(out_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))))
        return self.pool(x).squeeze(-1), x.transpose(1, 2)

class CrossAttentionFusion(nn.Module):
    def __init__(self, num_users=16, d_model=128, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        
        self.ppg_encoder = ResNetEncoder(in_channels=1, d_model=d_model)
        self.temp_encoder = ResNetEncoder(in_channels=1, d_model=d_model)
        
        self.cross_att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # Fusion (Addition)
        self.fusion_fc = nn.Linear(d_model, d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_users)
        )

    def forward(self, x_ppg, x_temp):
        z_p, seq_p = self.ppg_encoder(x_ppg)
        z_t, seq_t = self.temp_encoder(x_temp)
        
        q_p, q_t = z_p.unsqueeze(1), z_t.unsqueeze(1)
        
        att_p, _ = self.cross_att(q_p, seq_t, seq_t)
        z_p_r = self.norm_p(z_p + att_p.squeeze(1))
        
        att_t, _ = self.cross_att(q_t, seq_p, seq_p)
        z_t_r = self.norm_t(z_t + att_t.squeeze(1))
        
        # Addition Fusion
        z_fused = z_p_r + z_t_r
        z_fused = F.relu(self.fusion_fc(z_fused))
        
        return self.classifier(z_fused), z_p_r, z_t_r

# ==========================================
# 4. 손실 함수
# ==========================================
class AlignmentLoss(nn.Module):
    def __init__(self, t=0.07):
        super().__init__()
        self.t = t
        self.ce = nn.CrossEntropyLoss()
    def forward(self, z_a, z_b):
        logits = torch.matmul(F.normalize(z_a, dim=1), F.normalize(z_b, dim=1).T) / self.t
        return self.ce(logits, torch.arange(z_a.size(0)).to(z_a.device))

class SpreadControlLoss(nn.Module):
    def __init__(self, th=0.001):
        super().__init__()
        self.th = th
    def forward(self, z):
        return F.relu(torch.var(F.normalize(z, dim=1), dim=0).mean() - self.th)

# ==========================================
# 5. 실행 (Validation & Scheduler 추가)
# ==========================================
if __name__ == "__main__":
    data_folder = "./data/PPG_ECG_Data"
    
    # 1. 데이터 로드
    dataset = MultiModalDataset(data_folder, window_size=300, stride=50)
    
    if len(dataset) > 0:
        # 2. Train/Val 분할 (90:10)
        train_len = int(0.9 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        
        # DataLoader 분리
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"📊 분할 완료: Train {len(train_set)} / Val {len(val_set)}", flush=True)

        # 3. 모델 초기화 (ResNet Baseline + Addition)
        model = CrossAttentionFusion(num_users=NUM_USERS, d_model=D_MODEL, num_heads=NUM_HEADS).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=LR)
        # 스케줄러: Validation Loss가 정체되면 학습률 감소 (verbose 제거됨)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        crit_cls = nn.CrossEntropyLoss()
        crit_align = AlignmentLoss()
        crit_spread = SpreadControlLoss()

        print("🚀 학습 시작 (Validation 포함)...", flush=True)

        for epoch in range(EPOCHS):
            # --- Training Phase ---
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            # TQDM for training loop
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
            
            # [수정된 부분] 괄호 ( )를 사용하여 구조에 맞게 Unpacking
            for (ppg, temp), labels in pbar:
                ppg, temp, labels = ppg.to(DEVICE), temp.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                out, z_p, z_t = model(ppg, temp)
                
                loss = crit_cls(out, labels) + \
                       LAMBDA_A * (crit_align(z_p, z_t) + crit_align(z_t, z_p)) + \
                       LAMBDA_S * (crit_spread(z_p) + crit_spread(z_t))
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
                # 진행바 업데이트
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100*correct/total:.2f}%"})

            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * correct / total

            # --- Validation Phase ---
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                # 여기도 동일하게 괄호 ( ) 사용
                for (ppg, temp), labels in val_loader:
                    ppg, temp, labels = ppg.to(DEVICE), temp.to(DEVICE), labels.to(DEVICE)
                    out, _, _ = model(ppg, temp)
                    
                    # Val Loss는 Classification Loss만 봅니다 (성능 판단용)
                    loss = crit_cls(out, labels)
                    val_loss += loss.item()
                    
                    _, pred = torch.max(out.data, 1)
                    val_total += labels.size(0)
                    val_correct += (pred == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # 스케줄러 업데이트 (Val Loss 기준)
            scheduler.step(avg_val_loss)
            cur_lr = optimizer.param_groups[0]['lr']

            print(f"✨ [Epoch {epoch+1}/{EPOCHS}] LR: {cur_lr:.6f}")
            print(f"   Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"   Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%", flush=True)
            print("-" * 60, flush=True)

        print("🏁 모든 학습 완료!", flush=True)
