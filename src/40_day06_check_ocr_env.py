from pathlib import Path
import platform
import shutil
import subprocess

import cv2
import pytesseract


# ========================================
# 프로젝트 경로
# ========================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day06_ocr_environment.txt"
)


# ========================================
# 기본 환경 확인
# ========================================

lines = [
    f"OS: {platform.platform()}",
    f"OpenCV: {cv2.__version__}",
]


# ========================================
# Tesseract 실행 파일 확인
# ========================================

tesseract_path = shutil.which(
    "tesseract"
)

if tesseract_path is None:

    lines.append(
        "Tesseract: NOT FOUND"
    )

    lines.append(
        "Tesseract Path: NOT FOUND"
    )

else:

    lines.append(
        f"Tesseract Path: {tesseract_path}"
    )

    try:

        version = (
            pytesseract
            .get_tesseract_version()
        )

        lines.append(
            f"Tesseract: {version}"
        )

    except Exception as error:

        lines.append(
            f"Tesseract Version Error: {error}"
        )


    # ====================================
    # 설치된 OCR 언어 확인
    # ====================================

    try:

        result = subprocess.run(
            [
                "tesseract",
                "--list-langs",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        lines.append(
            "Languages:"
        )

        lines.append(
            result.stdout.strip()
        )

    except Exception as error:

        lines.append(
            f"Language Check Error: {error}"
        )


# ========================================
# Report 저장
# ========================================

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ========================================
# 결과 출력
# ========================================

print(
    "\n".join(lines)
)

print()

print(
    "Saved:",
    REPORT_PATH
)


if tesseract_path is None:

    print(
        "OCR Environment Check Failed"
    )

else:

    print(
        "OCR Environment OK"
    )