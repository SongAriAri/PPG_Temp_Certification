## Experiment Results

### SSM Encoder + Element-wise Sum Concat Fusion + Cross-Attention (baseline) 
- Accuracy : 25.72%
- (15 epoch)

### ResNet Encoder + Element-wise Sum Concat Fusion + Cross-Attention
- 방대한 양의 데이터 처리를 위해 ResNet의 Encoder로 교체
- Accuracy : 69.35%
- (40 epoch)

### ResNet Encoder + Concat Fusion + Cross-Attention
- embedding을 합치는 것이 아닌 옆으로 concat하는 방식
- Accuracy : 66.78%
- (20 epoch)

### ResNet Encoder (Large) + Element-wise Sum Concat Fusion + Cross-Attention
- ResNet의 Layer를 추가함 (더 깊게)
- Accuracy : 63.37%
- (30 epoch)

### ResNet Encoder (Large) + Concat Fusion + Cross-Attention
- Accuracy : 73.01%
- (40 epoch)

  

