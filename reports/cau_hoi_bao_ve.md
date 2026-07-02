# Câu hỏi bảo vệ và gợi ý trả lời

Tài liệu này hỗ trợ phần bảo vệ đồ án **Fruit Quality Inspection and Export Suitability Assessment using Traditional Image Processing and Machine Learning**. Các câu trả lời được viết ngắn gọn để nhóm có thể dùng khi thuyết trình hoặc trả lời giảng viên.

## 1. Tổng quan pipeline xử lý ảnh

### Câu hỏi: Pipeline chính của hệ thống gồm những bước nào?

Hệ thống gồm các bước: đọc ảnh, đánh giá chất lượng ảnh, tiền xử lý thích nghi, phân đoạn vùng quả, trích xuất đặc trưng, dự đoán bằng mô hình học máy, kiểm tra độ tin cậy, và đưa ra quyết định phù hợp xuất khẩu.

### Câu hỏi: Vì sao cần tách pipeline thành nhiều bước nhỏ?

Việc tách pipeline giúp từng bước dễ kiểm thử, dễ giải thích và dễ xác định nguyên nhân khi kết quả sai. Đây cũng là cách phù hợp với đồ án xử lý ảnh truyền thống.

### Câu hỏi: Điểm mạnh của pipeline truyền thống là gì?

Điểm mạnh là tính giải thích được. Nhóm có thể chỉ ra ảnh được biến đổi như thế nào, mask được tạo ra ra sao, đặc trưng nào được đưa vào mô hình, và vì sao hệ thống chọn kết quả cuối cùng.

## 2. Cài đặt bằng NumPy

### Câu hỏi: Vì sao dự án dùng NumPy cho xử lý ảnh lõi?

NumPy cho phép xử lý ảnh dưới dạng mảng số, phù hợp để tự cài đặt các phép toán như grayscale, lọc, histogram, thresholding, morphology và connected components.

### Câu hỏi: Vì sao không dùng OpenCV?

Dự án không dùng OpenCV để chứng minh nhóm hiểu thuật toán thay vì chỉ gọi hàm có sẵn. Điều này phù hợp với mục tiêu học thuật của môn xử lý ảnh và thị giác máy tính.

### Câu hỏi: PIL được dùng để làm gì?

PIL chỉ được dùng cho đọc ảnh, lưu ảnh và resize. Các phép xử lý ảnh chính vẫn được thực hiện bằng NumPy.

## 3. Tiền xử lý và phân đoạn

### Câu hỏi: Vì sao cần tiền xử lý thích nghi?

Ảnh trái cây có độ sáng, độ tương phản và nhiễu khác nhau. Tiền xử lý thích nghi giúp hệ thống chọn bước xử lý phù hợp với từng ảnh thay vì dùng một công thức cố định cho mọi trường hợp.

### Câu hỏi: Otsu thresholding hoạt động như thế nào?

Otsu tìm một ngưỡng cường độ sao cho hai nhóm pixel được tách ra tốt nhất. Ý tưởng là chọn ngưỡng làm tăng sự khác biệt giữa foreground và background.

### Câu hỏi: Vì sao cần morphology?

Morphology giúp loại bỏ nhiễu nhỏ, lấp lỗ trong mask và làm vùng quả liền mạch hơn. Điều này giúp bước trích xuất đặc trưng ổn định hơn.

### Câu hỏi: Connected components dùng để làm gì?

Connected components tìm các vùng liên thông trong ảnh nhị phân. Hệ thống dùng bước này để chọn vùng có khả năng là quả và bỏ qua các vùng nền nhỏ.

### Câu hỏi: Khi nào segmentation có thể sai?

Segmentation có thể sai khi nền quá phức tạp, ánh sáng quá gắt, quả bị che khuất, ảnh có nhiều quả, hoặc màu nền gần giống màu quả.

## 4. Trích xuất đặc trưng

### Câu hỏi: Hệ thống trích xuất những đặc trưng nào?

Hệ thống trích xuất đặc trưng hình dạng, thống kê màu RGB, histogram màu chuẩn hóa, thống kê grayscale, gradient, độ sáng, độ tương phản, mức nhiễu ước lượng, diện tích lỗi và tỷ lệ lỗi.

### Câu hỏi: Vì sao cần đặc trưng hình dạng?

Đặc trưng hình dạng giúp phân biệt các loại quả có cấu trúc khác nhau, ví dụ chuối thường dài hơn còn cam thường tròn hơn.

### Câu hỏi: Vì sao cần đặc trưng màu?

Màu sắc liên quan trực tiếp đến loại quả và độ tươi. Ví dụ vùng thâm, tối hoặc đổi màu có thể là dấu hiệu chất lượng kém.

### Câu hỏi: Defect ratio có ý nghĩa gì?

Defect ratio biểu diễn tỷ lệ vùng nghi ngờ lỗi trên vùng quả. Giá trị này hỗ trợ quyết định chất lượng và phân hạng thị trường.

## 5. Lựa chọn mô hình học máy

### Câu hỏi: Vì sao dùng KNN, SVM và Random Forest?

Ba mô hình này phù hợp với dữ liệu dạng bảng từ đặc trưng thủ công. KNN là baseline đơn giản, SVM xử lý ranh giới phân lớp tốt, còn Random Forest xử lý quan hệ phi tuyến và dễ dùng với nhiều đặc trưng.

### Câu hỏi: Vì sao không dùng deep learning?

Mục tiêu đồ án là giải thích pipeline xử lý ảnh truyền thống. Deep learning cần nhiều dữ liệu và tài nguyên hơn, đồng thời khó giải thích từng đặc trưng hơn trong phần bảo vệ.

### Câu hỏi: Hệ thống chọn mô hình tốt nhất như thế nào?

Hệ thống huấn luyện nhiều mô hình và so sánh bằng các chỉ số đánh giá. Mô hình có kết quả tốt nhất trên tập đánh giá được lưu để dự đoán.

## 6. Chỉ số đánh giá

### Câu hỏi: Dự án dùng những metric nào?

Dự án dùng accuracy, precision, recall, F1-score, classification report và confusion matrix để đánh giá mô hình.

### Câu hỏi: Vì sao không chỉ dùng accuracy?

Accuracy cho biết tỷ lệ đúng tổng thể nhưng không cho biết lớp nào bị nhầm nhiều. Precision, recall, F1-score và confusion matrix giúp phân tích lỗi chi tiết hơn.

### Câu hỏi: Kết quả hiện tại ở mức nào?

Theo các report đã lưu, độ chính xác nhận dạng loại quả khoảng `0.95`, còn chất lượng khoảng `0.94` đến `0.96` tùy cách đánh giá model riêng hoặc end-to-end.

## 7. Confidence và Need Recheck

### Câu hỏi: Vì sao cần confidence score?

Confidence score giúp hệ thống biết mức độ chắc chắn của dự đoán. Nếu mô hình không đủ chắc chắn, hệ thống không nên đưa ra quyết định xuất khẩu quá mạnh.

### Câu hỏi: Vì sao có trạng thái Need Recheck?

`Need Recheck` là cơ chế an toàn. Khi confidence thấp hoặc bằng chứng ảnh không khớp với dự đoán, hệ thống yêu cầu kiểm tra lại thay vì tự động chấp nhận hoặc loại bỏ.

### Câu hỏi: Model-evidence consistency check là gì?

Đây là bước kiểm tra xem dự đoán của mô hình có hợp lý với đặc trưng ảnh hay không. Nếu mô hình dự đoán tươi nhưng defect ratio hoặc bằng chứng chất lượng không phù hợp, hệ thống có thể hạ mức quyết định.

## 8. GUI

### Câu hỏi: GUI dùng để làm gì?

GUI giúp người dùng chọn ảnh, chạy dự đoán, xem kết quả, xem độ tin cậy và hiểu quyết định phân hạng mà không cần dùng dòng lệnh.

### Câu hỏi: GUI có thay đổi thuật toán không?

Không. GUI chỉ gọi pipeline đã có và hiển thị kết quả theo cách dễ hiểu hơn cho demo.

### Câu hỏi: Khi demo GUI nên trình bày gì?

Nhóm nên trình bày ảnh đầu vào, kết quả dự đoán, confidence, bằng chứng ảnh, export suitability và lý do nếu hệ thống trả về `Need Recheck`.

## 9. Hạn chế hiện tại

### Câu hỏi: Hạn chế chính của hệ thống là gì?

Hệ thống chỉ hỗ trợ một số loại quả, phụ thuộc vào chất lượng segmentation, và hoạt động tốt nhất với ảnh có một quả rõ ràng trên nền không quá phức tạp.

### Câu hỏi: Vì sao ảnh bên ngoài vẫn có thể fail?

Ảnh bên ngoài có thể khác tập huấn luyện về ánh sáng, nền, góc chụp, kích thước quả, độ nén ảnh hoặc có nhiều vật thể. Những yếu tố này làm segmentation và feature extraction khó hơn.

### Câu hỏi: Hệ thống đã sẵn sàng cho nhà máy chưa?

Chưa. Phiên bản hiện tại phù hợp cho demo học thuật. Để dùng trong nhà máy cần camera cố định, ánh sáng chuẩn, dữ liệu thực tế, kiểm thử tốc độ và tích hợp phần cứng loại bỏ sản phẩm.

## 10. Triển khai thực tế

### Câu hỏi: Nếu triển khai trên băng chuyền cần bổ sung gì?

Cần camera cố định, đèn chiếu ổn định, nền đồng nhất, cảm biến kích hoạt chụp ảnh, xử lý thời gian thực, lưu log kiểm định và cơ cấu phân loại sản phẩm.

### Câu hỏi: Cải tiến dữ liệu như thế nào?

Nhóm nên thu thêm ảnh từ môi trường thực tế, nhiều lô trái cây, nhiều mức hư hỏng, nhiều điều kiện ánh sáng và các trường hợp khó như quả bị che khuất hoặc có nhiều quả trong ảnh.

### Câu hỏi: Hướng phát triển tiếp theo là gì?

Hướng tiếp theo là mở rộng dataset, cải thiện segmentation, thêm đánh giá batch, tối ưu tốc độ, và so sánh với deep learning như một baseline nghiên cứu trong tương lai.
