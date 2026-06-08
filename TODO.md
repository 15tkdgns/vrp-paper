# VRP 논문 TODO

## 완료
- [x] E1. ch3.txt seed=0 → 42 (L117, L189)
- [x] E2. ch3.txt 실현변동성 → 실현분산 (L108)
- [x] E3. ch2.txt 4.5.3절 → Ⅳ.5.3절 (L18)
- [x] D-A. ch5.txt '안정적으로 작동' → Pooled R² 기준 + 90d+ R²_Within 단서
- [x] D-F. ch3.txt Pooled R² 정의 주석 (자산별 단순 평균 아님) 추가
- [x] appendix.txt 부록 B seed=0 → 42 (RF/XGB/LGB/MLP/BiLSTM-A)
- [x] appendix.txt 부록 B RF 수정: n_estimators 200 고정, max_depth {5,8,10}, min_samples_leaf {5,20} 추가
- [x] appendix.txt 부록 B MLP 수정: dropout → alpha(L2) {0.0001,0.01}, max_iter=500 추가
- [x] appendix.txt 부록 B BiLSTM-A 추가: optimizer=Adam, lr=0.001, batch_size=64, epochs 15/20

## 대기 중

### 복수 시드 실험
- [ ] 복수 시드 실험 코드 수정 명세 작성 (BiLSTM-A·MLP, seeds={0,1,2,3,42}, 22d 기준)
- [ ] 복수 시드 실험 Linux 환경(/root/vrp/)에서 실행
- [ ] 실험 결과 → ch5 한계 절 반영

### 최신 모형 비교 실험 (발표피드백 6번 대응)
- [ ] CatBoost 실험: 22d 기준 WEns·XGBoost·LightGBM과 비교, Linux 환경(/root/vrp/)에서 실행
- [ ] DLinear 실험: 22d 기준 Ridge·WEns와 비교, Linux 환경에서 실행
- [ ] PatchTST 실험: 22d 기준 BiLSTM-A·WEns와 비교, Linux 환경에서 실행
- [ ] 실험 결과 → ch5 향후 연구 방향 절 반영

### RepeatRV 업데이트 (발표피드백 7번 대응)
- [ ] RepeatRV baseline 재실험 (master_paper 수치 기준으로 통일)
- [ ] 발표피드백_답변.md 표 수치 업데이트 (RepeatRV 실험 완료 후)
