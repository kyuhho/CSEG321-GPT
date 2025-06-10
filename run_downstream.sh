#!/bin/bash

# 스크립트 실행 시작 시간 출력
echo "실험 스크립트 실행 시작: $(date)"
echo "============================================"

# 1. paraphrase_detection.py 실행 및 결과 저장
echo "1. paraphrase_detection.py 실행 중..."
python paraphrase_detection.py --epochs 1 --dataset_limit 2000 2>&1 | tee result_paraphrase_detection.txt
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "   ✓ paraphrase_detection.py 완료 - 결과는 result_paraphrase_detection.txt에 저장됨"
else
    echo "   ✗ paraphrase_detection.py 실행 중 오류 발생"
fi

echo ""

# 2. sonnet_generation.py 실행 및 결과 저장
echo "2. sonnet_generation.py 실행 중..."
python sonnet_generation.py --epochs 1 2>&1 | tee result_sonnet_generation.txt
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "   ✓ sonnet_generation.py 완료 - 결과는 result_sonnet_generation.txt에 저장됨"
else
    echo "   ✗ sonnet_generation.py 실행 중 오류 발생"
fi

echo ""

# 3. classifier.py 실행 및 결과 저장
echo "3. classifier.py 실행 중..."
python classifier.py --epochs 1 2>&1 | tee result_classifier.txt
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "   ✓ classifier.py 완료 - 결과는 result_classifier.txt에 저장됨"
else
    echo "   ✗ classifier.py 실행 중 오류 발생"
fi

echo ""
echo "============================================"
echo "모든 실험 완료: $(date)"
echo ""
echo "생성된 결과 파일들:"
echo "- result_paraphrase_detection.txt"
echo "- result_sonnet_generation.txt"
echo "- result_classifier.txt"