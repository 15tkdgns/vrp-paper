# VRP 연구 대시보드

Cross-Asset Volatility Risk Premium(VRP) 예측을 위한 머신러닝 프레임워크 연구 결과 대시보드입니다.

## 🚀 주요 특징
- **1일 예측 (V50)**: Dual-Attention LSTM 기반 (R² = 0.821)
- **22일 예측 (V36)**: 자산 적응형 Ridge 기반 (R² = 0.755)
- **트레이딩 전략**: Sharpe 1.84 달성

## 📁 프로젝트 구조
- `content/`: 메인 홈페이지 및 개요
- `experiments/`: 예측 지평별 모델 성능 및 검증 실험 결과
- `literature/`: 금융 및 CS 관련 문헌 리뷰 데이터
- `data/`: 실험에 사용된 변동성 데이터
- `src/`: 모델링 및 데이터 처리를 위한 핵심 코드

## 🛠️ 실행 방법 (Quarto)
```bash
# 로컬 프리뷰
quarto preview

# 정적 사이트 빌드
quarto render
```

---
*본 프로젝트는 Quarto를 사용하여 제작되었습니다.*
