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

# ==========================================
# 1. 환경 설정 (Hyperparameters)
# ==========================================
NUM_USERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [모델 파라미터]
D_MODEL = 128        # 특징 벡터 차원
NUM_HEADS = 4        # Attention Head 수
BATCH_SIZE = 128     # 배치 크기
LR = 0.001           # 학습률
EPOCHS = 30          # 총 학습 횟수

# [손실 함수 가중치]
LAMBDA_A = 0.5       
LAMBDA_S = 0.01      

print(f"🚀 [Start] PPG+Temp+Acc Authentication System (Fusion: CONCAT)", flush=True)
print(f"⚙️  Device: {DEVICE} | Split: Chronological (Time-based)", flush=True)

# ==========================================
# 2. 데이터셋 클래스 (기존 유지)
# ==========================================
class TriModalDataset(Dataset):
    def __init__(self, data_folder, window_size=300, stride=50, fs=128, mode='train', split_ratio=0.9):
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.mode = mode
        self.split_ratio = split_ratio
        
        self.samples_ppg = []
        self.samples_temp = []
        self.samples_acc = []  
        self.labels = []
        
        search_pattern = os.path.join(data_folder, "user_*.csv")
        file_list = glob.glob(search_pattern)
        
        if not file_list:
            print(f"❌ Error: 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {data_folder}")
            return

        print(f"📂 [{mode.upper()}] 데이터 로딩 중... (총 {len(file_list)}명)")
        
        for filepath in file_list:
            try:
                filename = os.path.basename(filepath)
                try:
                    user_num = int(filename.split('_')[1].split('.')[0])
                except:
                    user_num = int(filename.split('_')[1])

                df = pd.read_csv(filepath)
                df.columns = [c.strip() for c in df.columns]

                if user_num == 4:
                    df_segments = [df.iloc[:3786928], df.iloc[4194811:]]
                elif user_num == 6:
                    df_segments = [df.iloc[:4337569], df.iloc[4545544:]]
                else:
                    df_segments = [df]
                
                label = user_num - 1 
                required_cols = ['PPG', 'temperature', 'acc_x', 'acc_y', 'acc_z']
                
                user_ppg, user_temp, user_acc, user_lbl = [], [], [], []

                for segment_df in df_segments:
                    if segment_df.empty: continue
                    if not all(col in segment_df.columns for col in required_cols): continue

                    raw_ppg = segment_df['PPG'].values
                    raw_temp = segment_df['temperature'].values
                    raw_acc = segment_df[['acc_x', 'acc_y', 'acc_z']].values.T 

                    detrended = signal.detrend(raw_ppg)
                    b, a = signal.butter(4, [0.5/(0.5*fs), 8.0/(0.5*fs)], btype='band')
                    filtered = signal.filtfilt(b, a, detrended)
                    processed_ppg = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
                    
                    processed_temp = (raw_temp - 25.0) / (40.0 - 25.0) 

                    acc_mean = np.mean(raw_acc, axis=1, keepdims=True)
                    acc_std = np.std(raw_acc, axis=1, keepdims=True) + 1e-6
                    processed_acc = (raw_acc - acc_mean) / acc_std

                    num_windows = (len(processed_ppg) - window_size) // stride
                    if num_windows <= 0: continue

                    for i in range(num_windows):
                        start = i * stride
                        end = start + window_size
                        
                        user_ppg.append(processed_ppg[start:end])
                        user_temp.append(processed_temp[start:end])
                        user_acc.append(processed_acc[:, start:end])
                        user_lbl.append(label)

                total_len = len(user_ppg)
                split_idx = int(total_len * self.split_ratio)
                
                if self.mode == 'train':
                    self.samples_ppg.extend(user_ppg[:split_idx])
                    self.samples_temp.extend(user_temp[:split_idx])
                    self.samples_acc.extend(user_acc[:split_idx])
                    self.labels.extend(user_lbl[:split_idx])
                else:
                    self.samples_ppg.extend(user_ppg[split_idx:])
                    self.samples_temp.extend(user_temp[split_idx:])
                    self.samples_acc.extend(user_acc[split_idx:])
                    self.labels.extend(user_lbl[split_idx:])
                        
            except Exception as e:
                print(f"❌ 로드 에러 {filename}: {e}")

        self.samples_ppg = np.array(self.samples_ppg, dtype=np.float32)
        self.samples_temp = np.array(self.samples_temp, dtype=np.float32)
        self.samples_acc = np.array(self.samples_acc, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        if len(self.labels) > 0:
            self.samples_ppg = np.expand_dims(self.samples_ppg, axis=1)
            self.samples_temp = np.expand_dims(self.samples_temp, axis=1)

        print(f"🎉 [{mode.upper()}] 로드 완료! 샘플 수: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples_ppg[idx]), 
                torch.from_numpy(self.samples_temp[idx]),
                torch.from_numpy(self.samples_acc[idx])), torch.tensor(self.labels[idx])

# ==========================================
# 3. 모델 아키텍처 (Fusion 수정됨: Concat)
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

class TriModalFusion(nn.Module):
    def __init__(self, num_users=16, d_model=128, num_heads=4):
        super(TriModalFusion, self).__init__()
        # Encoders
        self.ppg_encoder = ResNetEncoder(in_channels=1, d_model=d_model)
        self.temp_encoder = ResNetEncoder(in_channels=1, d_model=d_model)
        self.acc_encoder = ResNetEncoder(in_channels=3, d_model=d_model) 
        
        # Cross Attention
        self.cross_att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        
        # --- [NEW] Fusion Layer (Concat Version) ---
        # 3개의 128차원 벡터를 Concat하므로 입력 차원은 d_model * 3 (384)
        self.fusion_projector = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, d_model)  # 다시 128로 압축
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_users)
        )

    def forward(self, x_ppg, x_temp, x_acc):
        # 1. Feature Extraction
        z_p, seq_p = self.ppg_encoder(x_ppg)
        z_t, seq_t = self.temp_encoder(x_temp)
        z_a, seq_a = self.acc_encoder(x_acc)
        
        # 2. Cross Attention
        att_p, _ = self.cross_att(z_p.unsqueeze(1), seq_t, seq_t)
        z_p_r = self.norm_p(z_p + att_p.squeeze(1))
        
        att_t, _ = self.cross_att(z_t.unsqueeze(1), seq_p, seq_p)
        z_t_r = self.norm_t(z_t + att_t.squeeze(1))
        
        z_a_r = self.norm_a(z_a)

        # 3. Fusion (Concatenation) ⭐ 수정된 부분 ⭐
        # [Batch, 128] 3개를 이어 붙여 [Batch, 384]로 만듦
        combined = torch.cat([z_p_r, z_t_r, z_a_r], dim=1) 
        
        # 4. Projection (압축)
        # [Batch, 384] -> [Batch, 128]
        z_fused = self.fusion_projector(combined)
        
        # 5. Normalization (L2 Norm)
        # 최종 임베딩을 구형 공간에 매핑 (Cosine Similarity 최적화)
        z_fused_final = F.normalize(z_fused, p=2, dim=1)
        
        # Return: Logits, Final Embedding, and Individual Embeddings
        return self.classifier(z_fused_final), z_fused_final, z_p_r, z_t_r, z_a_r

# ==========================================
# 4. 손실 함수 (Alignment & Spread)
# ==========================================
class AlignmentLoss(nn.Module):
    def __init__(self, t=0.07):
        super().__init__()
        self.t = t
        self.ce = nn.CrossEntropyLoss()
    def forward(self, z_a, z_b):
        logits = torch.matmul(F.normalize(z_a, dim=1), F.normalize(z_b, dim=1).T) / self.t
        labels = torch.arange(z_a.size(0)).to(z_a.device)
        return self.ce(logits, labels)

class SpreadControlLoss(nn.Module):
    def __init__(self, th=0.001):
        super().__init__()
        self.th = th
    def forward(self, z):
        return F.relu(torch.var(F.normalize(z, dim=1), dim=0).mean() - self.th)

# ==========================================
# 5. EER 계산 및 평가 함수
# ==========================================
def calculate_eer(genuine_scores, impostor_scores):
    scores = np.concatenate([genuine_scores, impostor_scores])
    # genuine은 점수가 높아야 하고, impostor는 낮아야 함
    
    thresholds = np.linspace(scores.min() - 0.01, scores.max() + 0.01, 1000)
    
    # FAR: 남인데 문 열어준 비율 (Threshold 이상인 Impostor)
    far = np.array([np.sum(impostor_scores >= t) / len(impostor_scores) for t in thresholds])
    # FRR: 본인인데 문 안 열어준 비율 (Threshold 미만인 Genuine)
    frr = np.array([np.sum(genuine_scores < t) / len(genuine_scores) for t in thresholds])
    
    diff = np.abs(far - frr)
    eer_idx = np.argmin(diff)
    
    return (far[eer_idx] + frr[eer_idx]) / 2

def generate_verification_scores(model, data_loader, device, num_users):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for (ppg, temp, acc), labels in tqdm(data_loader, desc="[Scoring]"):
            ppg, temp, acc = ppg.to(device), temp.to(device), acc.to(device)
            
            # 모델이 뱉어주는 'z_fused_final'을 직접 사용
            _, z_fused_final, _, _, _ = model(ppg, temp, acc)
            
            all_embeddings.append(z_fused_final.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    user_templates = {}
    for user_id in range(num_users):
        user_embs = all_embeddings[all_labels == user_id]
        if len(user_embs) > 0:
            user_templates[user_id] = np.mean(user_embs, axis=0)
        else:
            user_templates[user_id] = None 
    
    genuine_scores = []
    impostor_scores = []
    
    for emb, label in zip(all_embeddings, all_labels):
        target_template = user_templates.get(label)
        if target_template is None: continue

        sim_g = F.cosine_similarity(
            torch.from_numpy(emb).unsqueeze(0), 
            torch.from_numpy(target_template).unsqueeze(0)
        ).item()
        genuine_scores.append(sim_g)
        
        for other_id, other_template in user_templates.items():
            if other_id != label and other_template is not None:
                sim_i = F.cosine_similarity(
                    torch.from_numpy(emb).unsqueeze(0), 
                    torch.from_numpy(other_template).unsqueeze(0)
                ).item()
                impostor_scores.append(sim_i)

    return np.array(genuine_scores), np.array(impostor_scores)

# ==========================================
# 6. 메인 실행 루프
# ==========================================
if __name__ == "__main__":
    # 데이터 폴더 경로
    data_folder = "./data/Final_Data"
    
    train_dataset = TriModalDataset(data_folder, mode='train', split_ratio=0.9)
    val_dataset = TriModalDataset(data_folder, mode='val', split_ratio=0.9)
    
    if len(train_dataset) == 0:
        print("프로그램을 종료합니다.")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model = TriModalFusion(num_users=NUM_USERS, d_model=D_MODEL, num_heads=NUM_HEADS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    crit_cls = nn.CrossEntropyLoss()
    crit_align = AlignmentLoss()
    crit_spread = SpreadControlLoss()

    print(f"\n🔥 학습 시작 (총 {EPOCHS} Epochs)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for (ppg, temp, acc), labels in pbar:
            ppg, temp, acc, labels = ppg.to(DEVICE), temp.to(DEVICE), acc.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward: 반환값이 5개로 늘어남 (logits, z_final, z_p, z_t, z_a)
            out, z_final, z_p, z_t, z_a = model(ppg, temp, acc)
            
            # Loss Calculation
            loss_cls = crit_cls(out, labels)
            loss_align = crit_align(z_p, z_t) + crit_align(z_t, z_p) 
            loss_spread = crit_spread(z_p) + crit_spread(z_t) + crit_spread(z_a)
            
            total_loss = loss_cls + (LAMBDA_A * loss_align) + (LAMBDA_S * loss_spread)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}", 'Acc': f"{100*correct/total:.1f}%"})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for (ppg, temp, acc), labels in val_loader:
                ppg, temp, acc, labels = ppg.to(DEVICE), temp.to(DEVICE), acc.to(DEVICE), labels.to(DEVICE)
                
                # Validation에서는 z_final 등은 필요 없으므로 _ 처리
                out, _, _, _, _ = model(ppg, temp, acc)
                
                loss = crit_cls(out, labels)
                val_loss += loss.item()
                _, pred = torch.max(out.data, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        print(f"✨ [Ep {epoch+1}] Tr Loss: {avg_train_loss:.4f} (Acc {train_acc:.1f}%) | Val Loss: {avg_val_loss:.4f} (Acc {val_acc:.1f}%)")

    # 5. 최종 평가
    print("\n🏁 학습 종료. 최종 EER 계산 중...")
    gen_scores, imp_scores = generate_verification_scores(model, val_loader, DEVICE, NUM_USERS)
    
    if len(gen_scores) > 0:
        final_eer = calculate_eer(gen_scores, imp_scores)
        print(f"\n======================================")
        print(f"🏆 최종 결과 리포트 (Fusion: Concat)")
        print(f"   - Validation Accuracy : {val_acc:.2f}%")
        print(f"   - EER (Equal Error Rate): {final_eer * 100:.4f}%")
        print(f"======================================")
    else:
        print("⚠️ EER 계산 실패: 점수 데이터 부족")