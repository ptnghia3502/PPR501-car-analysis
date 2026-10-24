#!/bin/bash
# =============================================================================
# run_app.sh - Chạy app Streamlit trên Mac/Linux
# =============================================================================
# Cách dùng:
#   chmod +x run_app.sh
#   ./run_app.sh
# =============================================================================

cd "$(dirname "$0")"
echo "Đang chạy app tại: $(pwd)"
echo "Mở trình duyệt tại: http://localhost:8501"
echo "Nhấn Ctrl+C để dừng app."
echo ""
py -m streamlit run app.py
