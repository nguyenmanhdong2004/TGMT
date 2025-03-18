# class_names = {
# 0: "Tran_Minh_Thanh",
# 1: "Phan_Viet_Hung",
# 2: "Trieu_Quoc_Anh",
# 3: "Tran_Truong_Giang",
# 4: "Tran_Duc_Manh",
# 5: "Ngo_Nguyen_Quang_Linh",
# 6:"Nguyen_Manh_Duy",
# 7: "Le_Thi_Ngoc_Bich",
# 8: "Nguyen_Van_Dat",
# 9: "Nguyen_Van_Huy",
# 10: "Nguyen_Hoa_Binh",
# 11: "Do_Ngoc_Trung",
# 12: "Nguyen_Manh_Dong",
# 13: "Nguyen_Nam_Cuong",
# 14:"Nguyen_Trung_Thanh",
# 15: "Le_Trong_Thanh_Tung"}

import cv2
from ultralytics import YOLO

# Load mô hình đã huấn luyện
model = YOLO("runs/detect/train/weights/best.pt")  # Thay "best.pt" bằng mô hình của bạn

# Danh sách tên người từ file data.yaml
class_names = {
0: "Tran_Minh_Thanh",
1: "Phan_Viet_Hung",
2: "Trieu_Quoc_Anh",
3: "Tran_Truong_Giang",
4: "Tran_Duc_Manh",
5: "Ngo_Nguyen_Quang_Linh",
6:"Nguyen_Manh_Duy",
7: "Le_Thi_Ngoc_Bich",
8: "Nguyen_Van_Dat",
9: "Nguyen_Van_Huy",
10: "Nguyen_Hoa_Binh",
11: "Do_Ngoc_Trung",
12: "Nguyen_Manh_Dong",
13: "Nguyen_Nam_Cuong",
14:"Nguyen_Trung_Thanh",
15: "Le_Trong_Thanh_Tung"}

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện khuôn mặt
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ hộp
            cls_id = int(box.cls[0].item())  # ID lớp
            person_name = class_names.get(cls_id, "Unknown")  # Tìm tên theo ID

            # Vẽ hộp nhận diện
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Hiển thị tên người
            cv2.putText(frame, person_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị video với kết quả nhận diện
    cv2.imshow("Face Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
