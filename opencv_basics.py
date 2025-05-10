import cv2
import numpy as np

def basic_operations():
    # 读取图片
    img = cv2.imread('test.jpg')
    if img is None:
        print("无法读取图片，请确保图片路径正确")
        return
    
    # 显示图片
    cv2.imshow('原始图片', img)
    cv2.waitKey(0)
    
    # 图片基本属性
    print(f"图片尺寸: {img.shape}")
    print(f"图片类型: {img.dtype}")
    
    # 图片缩放
    resized = cv2.resize(img, (300, 300))
    cv2.imshow('缩放后的图片', resized)
    cv2.waitKey(0)
    
    # 图片旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('旋转后的图片', rotated)
    cv2.waitKey(0)
    
    # 图片灰度转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('灰度图片', gray)
    cv2.waitKey(0)
    
    # 图片模糊处理
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow('模糊处理后的图片', blurred)
    cv2.waitKey(0)
    
    # 边缘检测
    edges = cv2.Canny(gray, 100, 200)
    cv2.imshow('边缘检测', edges)
    cv2.waitKey(0)
    
    # 保存处理后的图片
    cv2.imwrite('processed_image.jpg', edges)
    
    # 关闭所有窗口
    cv2.destroyAllWindows()

def video_operations():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法获取画面")
            break
            
        # 显示原始画面
        cv2.imshow('摄像头画面', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("开始演示OpenCV基本操作...")
    basic_operations()
    print("\n开始演示视频操作...")
    video_operations() 