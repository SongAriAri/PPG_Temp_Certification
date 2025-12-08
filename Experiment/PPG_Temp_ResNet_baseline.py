import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal

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


import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. ResNet Backbone (Feature Extractor)
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
    def __init__(self, in_channels=1, d_model=128): # d_model을 128로 상향
        super(ResNetEncoder, self).__init__()
        
        # Initial Conv
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, d_model, 2, stride=2) 
        # Layer 4 제거 또는 조정 (시퀀스 길이가 너무 줄어들지 않도록)
        
        # Global Pooling (나중에 사용)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_planes, out_planes, blocks, stride):
        layers = []
        layers.append(BasicBlock1D(in_planes, out_planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, 300)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x) # (B, 32, 75)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # (B, 128, 19) -> 시퀀스 길이가 19로 줄어듦
        
        # Cross-Attention을 위해 (B, Length, Channel) 형태로 변환
        # (B, 128, 19) -> (B, 19, 128)
        seq_out = x.transpose(1, 2)
        
        # Global Pooling for Vector representation
        pooled_out = self.pool(x).squeeze(-1) # (B, 128)
        
        return pooled_out, seq_out

# ==========================================
# 2. ResNet + Cross-Attention Fusion Model
# ==========================================
class ResNetFusionModel(nn.Module):
    def __init__(self, num_users=16, d_model=128, num_heads=4):
        super(ResNetFusionModel, self).__init__()
        
        # ResNet Backbone 사용 (PPG & Temp 각각)
        self.ppg_encoder = ResNetEncoder(in_channels=1, d_model=d_model)
        self.temp_encoder = ResNetEncoder(in_channels=1, d_model=d_model)
        
        # Cross-Modal Attention (논문 구조 유지)
        self.cross_att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # Fusion & Classifier
        self.fusion_fc = nn.Linear(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_users)
        )

    def forward(self, x_ppg, x_temp):
        # 1. Encoding (ResNet)
        z_p, seq_p = self.ppg_encoder(x_ppg) # z: (B, 128), seq: (B, 19, 128)
        z_t, seq_t = self.temp_encoder(x_temp)
        
        # 2. Cross-Attention
        # Query는 Global Vector(z)를 시퀀스화해서 사용: (B, 1, 128)
        query_p = z_p.unsqueeze(1)
        query_t = z_t.unsqueeze(1)
        
        # Key, Value는 시퀀스 전체(seq)를 사용하여 디테일한 정보를 참조
        # PPG가 Temperature의 전체 흐름(seq_t)을 참조
        attn_out_p, _ = self.cross_att(query_p, seq_t, seq_t) 
        z_p_refined = self.norm_p(z_p + attn_out_p.squeeze(1))
        
        # Temperature가 PPG의 전체 흐름(seq_p)을 참조
        attn_out_t, _ = self.cross_att(query_t, seq_p, seq_p)
        z_t_refined = self.norm_t(z_t + attn_out_t.squeeze(1))
        
        # 3. Fusion
        z_fused = z_p_refined + z_t_refined
        z_fused = F.relu(self.fusion_fc(z_fused))
        
        # 4. Classification
        logits = self.classifier(z_fused)
        
        return logits, z_p_refined, z_t_refined
    

# ==========================================
# 3. 전체 모델 (Fusion & Classifier)
# ==========================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, num_users=16, d_model=64, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        
        # 듀얼 인코더 (PPG용, Temp용)
        self.ppg_encoder = ResNetEncoder(d_model=d_model)
        self.temp_encoder = ResNetEncoder(d_model=d_model)
        
        # Cross-Modal Attention
        self.cross_att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # Fusion & Classifier
        self.fusion_fc = nn.Linear(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_users)
        )

    def forward(self, x_ppg, x_temp):
        # 1. 인코딩
        z_p, seq_p = self.ppg_encoder(x_ppg) # z: (B, D), seq: (B, L, D)
        z_t, seq_t = self.temp_encoder(x_temp)
        
        # 2. Cross-Attention을 위한 차원 조정 (B, 1, D)
        query_p = z_p.unsqueeze(1)
        query_t = z_t.unsqueeze(1)
        
        # PPG가 Temp를 참조하여 보정
        attn_out_p, _ = self.cross_att(query_p, query_t, query_t)
        z_p_refined = self.norm_p(z_p + attn_out_p.squeeze(1))
        
        # Temp가 PPG를 참조하여 보정
        attn_out_t, _ = self.cross_att(query_t, query_p, query_p)
        z_t_refined = self.norm_t(z_t + attn_out_t.squeeze(1))
        
        # 3. Fusion (Element-wise Addition)
        z_fused = z_p_refined + z_t_refined
        z_fused = F.relu(self.fusion_fc(z_fused))
        
        # 4. Classification
        logits = self.classifier(z_fused)
        
        return logits, z_p_refined, z_t_refined
    

class AlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(AlignmentLoss, self).__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, z_a, z_b):
        # z_a, z_b: (Batch, D)
        # Cosine Similarity 계산
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)
        
        # (B, D) @ (D, B) -> (B, B) 유사도 행렬
        logits = torch.matmul(z_a, z_b.T) / self.temperature
        
        # 정답: 대각선 요소 (자기 자신과의 쌍)
        labels = torch.arange(z_a.size(0)).to(z_a.device)
        
        return self.ce_loss(logits, labels)

class SpreadControlLoss(nn.Module):
    def __init__(self, threshold=0.001):
        super(SpreadControlLoss, self).__init__()
        self.threshold = threshold

    def forward(self, z):
        # 특징 벡터들의 분산이 너무 커지지 않도록 제어
        z = F.normalize(z, dim=1)
        var = torch.var(z, dim=0).mean()
        # 분산이 threshold보다 크면 페널티
        loss = F.relu(var - self.threshold)
        return loss


import torch.optim as optim
from tqdm.auto import tqdm  # Jupyter/Console 자동 감지

# ==========================================
# 설정 (Hyperparameters)
# ==========================================
NUM_USERS = 16  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_A = 0.5  
LAMBDA_S = 0.01 
LR = 0.001
EPOCHS = 30   

print(f"🚀 학습 장치: {DEVICE}")

# ==========================================
# 모델 및 학습 준비
# ==========================================
model = CrossAttentionFusion(num_users=NUM_USERS).to(DEVICE)

criterion_cls = nn.CrossEntropyLoss()
criterion_align = AlignmentLoss()
criterion_spread = SpreadControlLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 학습 실행
# ==========================================
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total_samples = 0
    
    # DataLoader를 tqdm으로 감싸서 진행률 바 생성
    # desc: 바 왼쪽에 표시될 설명 (Epoch 번호)
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
    
    for batch_idx, ((ppg, temp), labels) in enumerate(progress_bar):
        # 데이터 GPU로 이동
        ppg = ppg.to(DEVICE)
        temp = temp.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # 1. Forward
        outputs, z_p, z_t = model(ppg, temp)
        
        # 2. Loss 계산
        loss_cls = criterion_cls(outputs, labels)
        
        # 정렬 손실 (Alignment: 양방향)
        loss_align = criterion_align(z_p, z_t) + criterion_align(z_t, z_p)
        
        # 분산 제어 손실 (Spread)
        loss_spread = criterion_spread(z_p) + criterion_spread(z_t)
        
        # 최종 Loss 합산
        loss = loss_cls + (LAMBDA_A * loss_align) + (LAMBDA_S * loss_spread)
        
        # 3. Backward & Update
        loss.backward()
        optimizer.step()
        
        # 통계 업데이트
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 실시간 정확도 계산
        current_acc = 100 * correct / total_samples
        
        # TQDM 바 우측에 실시간 정보 표시 (Loss, Accuracy)
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Acc': f"{current_acc:.2f}%",
            'Cls': f"{loss_cls.item():.4f}" # 분류 Loss만 따로 보고 싶다면 추가
        })

    # 에폭 종료 후 평균 기록 출력
    avg_loss = total_loss / len(dataloader)
    final_acc = 100 * correct / total_samples
    print(f"✨ Epoch {epoch+1} Summary - Avg Loss: {avg_loss:.4f}, Accuracy: {final_acc:.2f}%")

print("🏁 모든 학습 완료!")