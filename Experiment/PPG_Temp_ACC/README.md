## Experiment Results
(30 epoch로 통일)
______________________________________________________
### ResNet Encoder + Element-wise Sum Concat Fusion + Cross-Attention
- Best Validation Accuracy : 93.5%
- EER (Equal Error Rate): 3.0167%

### ResNet Encoder + Concat Fusion + Cross-Attention
embedding을 합치는 것이 아닌 옆으로 concat하는 방식
- Best Validation Accuracy : 93.4%
- EER (Equal Error Rate): 2.1413%

### ResNet Encoder (Large) + Element-wise Sum Concat Fusion + Cross-Attention
ResNet의 Layer를 추가함 (더 깊게)
- Best Validation Accuracy : 95.2%
- EER (Equal Error Rate): 4.1005%

### ResNet Encoder (Large) + Concat Fusion + Cross-Attention
- Best Validation Accuracy : 94.6%
- EER (Equal Error Rate): 2.0278%
