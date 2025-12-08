import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from scipy import signal

from tqdm.auto import tqdm


class MultiModalDataset(Dataset):
    def __init__(self, data_folder, window_size=300, stride=50, fs=128):
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        
        self.samples_ppg = []
        self.samples_temp = []
        self.labels = []
        
        # 1. 파일 리스트 로드
        search_pattern = os.path.join(data_folder, "user_*.csv")
        file_list = glob.glob(search_pattern)
        
        if not file_list:
            print(f"❌ 경고: '{data_folder}' 경로에서 파일을 하나도 찾지 못했습니다.")
            return

        print(f"📂 데이터 로딩 시작... (총 {len(file_list)}개 파일 감지)")
        
        for filepath in file_list:
            filename = os.path.basename(filepath)
            
            # 2. User ID 추출 (파일명 파싱)
            try:
                # user_4_part1_final.csv -> user, 4, part1, final.csv
                parts = filename.split('_')
                user_num = int(parts[1]) # 4 추출
                label = user_num - 1     # 0-based index
            except Exception as e:
                print(f"⚠️ 파일명 파싱 실패 ({filename}): {e}")
                continue

            # 3. CSV 읽기 및 전처리
            try:
                df = pd.read_csv(filepath)
                
                # [수정 완료] 확인된 컬럼명 직접 사용 ('Index', 'PPG', 'temperature')
                # 혹시 모를 공백 제거를 위해 컬럼명 strip 처리
                df.columns = [c.strip() for c in df.columns]
                
                if 'PPG' not in df.columns or 'temperature' not in df.columns:
                     print(f"  ❌ 컬럼 누락 ({filename}): {df.columns}")
                     continue

                raw_ppg = df['PPG'].values
                raw_temp = df['temperature'].values
                
                # 전처리: Detrending -> 4Hz Low-pass Filter
                processed_ppg = self.preprocess_ppg(raw_ppg)
                
                # 온도 정규화 (Min-Max Scaling: 25~40도 기준)
                processed_temp = (raw_temp - 25.0) / (40.0 - 25.0) 

                # 4. 슬라이딩 윈도우
                num_windows = (len(processed_ppg) - self.window_size) // self.stride
                
                if num_windows <= 0:
                    continue

                file_samples = 0
                for i in range(num_windows):
                    start = i * self.stride
                    end = start + self.window_size
                    
                    ppg_window = processed_ppg[start:end]
                    temp_window = processed_temp[start:end]
                    
                    # (Optional) 이상치 제거 (PPG Z-score > 5)
                    if np.max(np.abs(ppg_window)) > 5:
                        continue
                        
                    self.samples_ppg.append(ppg_window)
                    self.samples_temp.append(temp_window)
                    self.labels.append(label)
                    file_samples += 1
                
                print(f"  ✅ User {user_num} (Label {label}): {file_samples} windows loaded.")
                
            except Exception as e:
                print(f"  ❌ 데이터 로드 에러 ({filename}): {e}")

        # 리스트 -> Numpy -> Tensor
        self.samples_ppg = np.array(self.samples_ppg, dtype=np.float32)
        self.samples_temp = np.array(self.samples_temp, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # 차원 확장: (N, 1, Length)
        if len(self.labels) > 0:
            self.samples_ppg = np.expand_dims(self.samples_ppg, axis=1)
            self.samples_temp = np.expand_dims(self.samples_temp, axis=1)

        print(f"🎉 전체 데이터 로드 완료! 총 샘플 수: {len(self.labels)}")

    def preprocess_ppg(self, signal_data):
        # 1. Detrending
        detrended = signal.detrend(signal_data)
        # 2. Low-pass Filter (4Hz)
        nyquist = 0.5 * self.fs
        cutoff = 4.0 / nyquist
        b, a = signal.butter(4, cutoff, btype='low')
        filtered = signal.filtfilt(b, a, detrended)
        # 3. Z-score Normalization
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
        return normalized

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples_ppg[idx]), 
                torch.from_numpy(self.samples_temp[idx])), torch.tensor(self.labels[idx])
    


if __name__ == "__main__":
    data_folder = "./data/PPG_ECG_Data"
    
    # 윈도우 크기 300 (논문 기준), 스트라이드 50 (겹쳐서 데이터 증강 효과)
    dataset = MultiModalDataset(data_folder, window_size=300, stride=50)
    
    # DataLoader 생성
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 데이터 형상 확인
    if len(dataset) > 0:
        (ppg, temp), label = next(iter(dataloader))
        print("\n--- Batch Shape Check ---")
        print(f"PPG Input Shape : {ppg.shape}")   # (64, 1, 300)
        print(f"Temp Input Shape: {temp.shape}")  # (64, 1, 300)
        print(f"Label Shape     : {label.shape}") # (64,)


# ==========================================
# 1. 설정 (Hyperparameters)
# ==========================================
NUM_USERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 기존 ResNet 설정 유지하되, Concat을 위해 d_model만 살짝 키움
D_MODEL = 256       
NUM_HEADS = 8       
BATCH_SIZE = 128    
LR = 0.001          
EPOCHS = 10         

LAMBDA_A = 1.0      # 정렬 Loss 가중치 (Concat 시 중요)
LAMBDA_S = 0.005    

print(f"🚀 학습 장치: {DEVICE} | Concat Fusion 모드")



# ==========================================
# 2. 모델 아키텍처 (ResNet + Concat)
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

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=1, d_model=256):
        super(ResNetEncoder, self).__init__()
        # 초기 Conv
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Blocks (기존 3단계 유지)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, d_model, 2, stride=2)
        
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


class ConcatFusionModel(nn.Module):
    def __init__(self, num_users=16, d_model=256, num_heads=8):
        super(ConcatFusionModel, self).__init__()
        
        self.ppg_encoder = ResNetEncoder(in_channels=1, d_model=d_model)
        self.temp_encoder = ResNetEncoder(in_channels=1, d_model=d_model)
        
        self.cross_att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # [핵심 변경] Concat Fusion Layer
        # 입력이 (PPG + Temp)라서 2배가 됨 -> 다시 d_model로 압축
        self.fusion_fc = nn.Linear(d_model * 2, d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_users)
        )

    def forward(self, x_ppg, x_temp):
        # 1. Encoding
        z_p, seq_p = self.ppg_encoder(x_ppg)
        z_t, seq_t = self.temp_encoder(x_temp)
        
        query_p = z_p.unsqueeze(1)
        query_t = z_t.unsqueeze(1)
        
        # 2. Cross Attention
        attn_out_p, _ = self.cross_att(query_p, seq_t, seq_t)
        z_p_refined = self.norm_p(z_p + attn_out_p.squeeze(1))
        
        attn_out_t, _ = self.cross_att(query_t, seq_p, seq_p)
        z_t_refined = self.norm_t(z_t + attn_out_t.squeeze(1))
        
        # 3. [Concat Fusion 적용]
        # (B, D)와 (B, D)를 붙여서 (B, 2D)로 만듦
        z_concat = torch.cat([z_p_refined, z_t_refined], dim=1) 
        
        # FC Layer를 통과시켜 융합
        z_fused = F.relu(self.fusion_fc(z_concat))
        
        # 4. Classification
        logits = self.classifier(z_fused)
        
        return logits, z_p_refined, z_t_refined
    

# ==========================================
# 3. 손실 함수 (Loss Functions)
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
# 4. 학습 루프 (tqdm 제거, epoch 로그만 출력)
# ==========================================
# 데이터 로더 확인
if 'dataloader' not in locals():
    print("⚠️ datalo더가 정의되지 않았습니다. 이전 데이터 로드 코드를 먼저 실행해주세요!")
else:
    model = ConcatFusionModel(num_users=NUM_USERS, d_model=D_MODEL, num_heads=NUM_HEADS).to(DEVICE)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_align = AlignmentLoss()
    criterion_spread = SpreadControlLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🚀 학습 시작...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        # tqdm 제거 → 깔끔한 로그
        for (ppg, temp), labels in dataloader:
            ppg, temp, labels = ppg.to(DEVICE), temp.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, z_p, z_t = model(ppg, temp)
            
            # Loss 계산
            loss_cls = criterion_cls(outputs, labels)
            loss_align = criterion_align(z_p, z_t) + criterion_align(z_t, z_p)
            loss_spread = criterion_spread(z_p) + criterion_spread(z_t)
            
            loss = loss_cls + (LAMBDA_A * loss_align) + (LAMBDA_S * loss_spread)
            
            loss.backward()
            optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        acc = 100 * correct / total_samples
        
        # 👉 epoch당 한 줄만 출력
        print(f"[Epoch {epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    print("🏁 학습 완료!")