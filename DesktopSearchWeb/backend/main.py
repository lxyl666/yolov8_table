import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
from detector import YOLODetector

# 初始化FastAPI
app = FastAPI(title="桌面小物件识别系统", version="1.0")

# 允许跨域（前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化检测器
MODEL_PATH = r"D:\Graduation_Design_YOLO_AR\runs\detect\04_Model_Comparison_yolov8n\weights\best.onnx"
detector = YOLODetector(MODEL_PATH)

# 创建临时文件夹
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# 挂载静态文件（用于访问结果图片）
app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get("/")
async def root():
    return {"message": "桌面小物件识别系统API已启动"}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    上传图片检测，返回带框的图片URL和检测结果
    """
    # 保存上传的图片
    upload_path = UPLOAD_DIR / file.filename
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 检测
    result_img, detections = detector.detect_image(str(upload_path))

    # 保存结果图片
    result_filename = f"result_{file.filename}"
    result_path = RESULT_DIR / result_filename
    cv2.imwrite(str(result_path), result_img)

    return {
        "success": True,
        "result_image_url": f"/results/{result_filename}",
        "detections": detections,
        "count": len(detections)
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 启动Web服务: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)