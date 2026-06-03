#!/bin/bash
# Quarto 서버 실행 스크립트

echo "Quarto 연구노트 대시보드 서버를 시작합니다..."
echo ""

# 현재 디렉토리 확인
echo "현재 디렉토리: $(pwd)"
echo ""

# Quarto 설치 확인
if ! command -v quarto &> /dev/null; then
    echo "❌ Quarto가 설치되지 않았습니다."
    echo "설치 방법: https://quarto.org/docs/get-started/"
    exit 1
fi

echo "✅ Quarto 버전: $(quarto --version)"
echo ""

# Python 패키지 확인
echo "Python 패키지 확인 중..."
python3 -c "import pandas, numpy, matplotlib, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  일부 Python 패키지가 없을 수 있습니다."
    echo "설치: pip install pandas numpy matplotlib seaborn jupyter ipykernel"
    echo ""
fi

# 서버 실행
echo "🚀 서버를 시작합니다..."
echo "브라우저에서 http://localhost:4200 으로 접속하세요."
echo "종료하려면 Ctrl+C를 누르세요."
echo ""

quarto preview --port 4200
