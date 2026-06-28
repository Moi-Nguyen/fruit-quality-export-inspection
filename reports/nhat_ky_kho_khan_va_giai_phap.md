# Nhật ký khó khăn và giải pháp

## 1. Mục đích của tài liệu

Tài liệu này ghi lại quá trình phát triển dự án, các khó khăn kỹ thuật đã gặp, hiện tượng quan sát được, phân tích nguyên nhân, hướng giải quyết, quyết định triển khai và bài học rút ra. Nội dung trong tài liệu giúp nhóm nhìn lại quá trình làm việc một cách có hệ thống, đồng thời hỗ trợ viết báo cáo cuối kỳ, chuẩn bị slide thuyết trình và trả lời câu hỏi khi bảo vệ đồ án.

Mục tiêu của tài liệu không phải là trình bày kết quả theo hướng phóng đại, mà tập trung vào tư duy kỹ thuật và quá trình cải tiến dần pipeline xử lý ảnh. Đặc biệt, phần segmentation trong dự án được xem là bước ước lượng vùng quả để phục vụ trích xuất đặc trưng, không phải bài toán tạo ground-truth mask hoàn hảo ở mức từng pixel.

## 2. Tổng quan pipeline hiện tại

Pipeline hiện tại của dự án gồm các bước chính sau:

1. **Dataset sampling**: lấy mẫu cân bằng từ dataset gốc để tạo tập dữ liệu nhỏ hơn, phù hợp cho kiểm thử nhanh và quản lý trong quá trình phát triển.
2. **Image loading**: đọc ảnh đầu vào bằng PIL để chuyển dữ liệu ảnh sang dạng mảng số.
3. **RGB to grayscale**: chuyển ảnh màu RGB sang ảnh xám để phục vụ các bước phân tích độ sáng, tương phản và một số thuật toán ngưỡng.
4. **Image quality analysis**: phân tích chất lượng ảnh thông qua độ sáng, độ tương phản và mức nhiễu.
5. **Adaptive preprocessing**: chọn phương pháp tiền xử lý phù hợp với từng ảnh dựa trên kết quả phân tích chất lượng.
6. **Segmentation**: ước lượng vùng quả bằng các kỹ thuật ngưỡng, khoảng cách màu nền, loại bỏ viền đen, morphology và connected components.
7. **Các bước tiếp theo**: trích xuất đặc trưng, huấn luyện mô hình machine learning, đánh giá mức độ phù hợp xuất khẩu và xây dựng GUI.

## 3. Các khó khăn đã gặp và cách giải quyết

### Vấn đề 1: Dataset gốc quá lớn

**Hiện tượng quan sát được:**

Dataset gốc từ Kaggle có hàng nghìn ảnh thuộc nhiều lớp trái cây khác nhau. Nếu sử dụng toàn bộ dataset trong quá trình phát triển ban đầu, thời gian chạy thử sẽ lâu, việc kiểm tra lỗi chậm hơn và dung lượng dữ liệu không phù hợp để đưa trực tiếp lên GitHub.

**Nguyên nhân phân tích:**

Ở giai đoạn phát triển pipeline, nhóm cần kiểm thử nhiều lần sau mỗi thay đổi thuật toán. Việc xử lý toàn bộ dataset gây tốn thời gian không cần thiết. Ngoài ra, dữ liệu ảnh thường có dung lượng lớn, không nên quản lý trực tiếp bằng Git vì làm repository nặng và khó chia sẻ.

**Ý tưởng giải quyết:**

Giữ nguyên dataset gốc trong thư mục `data/raw`, sau đó tạo một tập mẫu cân bằng trong `data/sample` để dùng cho phát triển và kiểm thử nhanh. Việc lấy mẫu cần có random seed cố định để kết quả có thể tái lập.

**Cách đã triển khai:**

Nhóm tạo tập sample với random seed `42`. Với mỗi lớp, tập `train` lấy 150 ảnh và tập `test` lấy 50 ảnh. Các file dataset được cấu hình để Git bỏ qua, tránh đưa ảnh dữ liệu lên repository.

**Kết quả sau khi cải tiến:**

Thời gian chạy thử nhanh hơn, cấu trúc dữ liệu rõ ràng hơn và repository gọn hơn. Nhóm vẫn giữ được tính cân bằng tương đối giữa các lớp trong tập sample, thuận lợi cho kiểm thử pipeline.

**Bài học rút ra:**

Trong các dự án xử lý ảnh, không nhất thiết phải dùng toàn bộ dataset ngay từ đầu. Một tập mẫu nhỏ, cân bằng và có thể tái lập giúp phát triển thuật toán hiệu quả hơn.

### Vấn đề 2: Không được dùng OpenCV cho xử lý ảnh lõi

**Hiện tượng quan sát được:**

Dự án yêu cầu nhóm tự triển khai các thuật toán xử lý ảnh lõi thay vì sử dụng trực tiếp các hàm có sẵn từ OpenCV. Điều này làm cho một số thao tác như lọc ảnh, cân bằng histogram, thresholding, morphology và connected components mất nhiều thời gian hơn.

**Nguyên nhân phân tích:**

Nếu dùng OpenCV cho các bước xử lý chính, phần cài đặt sẽ ngắn hơn nhưng nhóm khó thể hiện rõ hiểu biết về nguyên lý thuật toán. Với đồ án thị giác máy tính ở năm ba, việc tự triển khai giúp chứng minh nhóm hiểu cách hoạt động của từng bước trong pipeline.

**Ý tưởng giải quyết:**

Chỉ dùng PIL cho việc đọc và lưu ảnh. Các thao tác xử lý ảnh lõi được triển khai bằng NumPy để đảm bảo nhóm vẫn kiểm soát được thuật toán và có thể giải thích khi báo cáo.

**Cách đã triển khai:**

PIL được sử dụng ở mức nhập/xuất dữ liệu ảnh. NumPy được dùng cho các phần xử lý chính như chuyển ảnh, phân tích thống kê, lọc, tính ngưỡng, xử lý mask và các phép toán trên mảng.

**Kết quả sau khi cải tiến:**

Mã nguồn dài hơn so với khi dùng thư viện có sẵn, nhưng pipeline minh bạch hơn. Nhóm có thể giải thích từng bước xử lý và lý do chọn thuật toán.

**Bài học rút ra:**

Tự triển khai thuật toán giúp hiểu sâu hơn về xử lý ảnh, dù cần nhiều công sức hơn. Đây là điểm phù hợp với mục tiêu học thuật của đồ án.

### Vấn đề 3: Ảnh có độ sáng, tương phản và nhiễu khác nhau

**Hiện tượng quan sát được:**

Một số ảnh trong dataset bị tối, một số ảnh có độ tương phản thấp, trong khi một số ảnh khác có nhiễu hoặc chi tiết nền làm ảnh hưởng đến segmentation. Nếu xử lý tất cả ảnh giống nhau, kết quả không ổn định.

**Nguyên nhân phân tích:**

Dataset được thu thập hoặc tăng cường từ nhiều điều kiện khác nhau. Ánh sáng, nền, nén ảnh và phép biến đổi dữ liệu làm cho đặc điểm ảnh thay đổi đáng kể giữa các mẫu.

**Ý tưởng giải quyết:**

Trước khi tiền xử lý, cần phân tích chất lượng ảnh để biết ảnh đang có vấn đề chính là tối, thiếu tương phản hay nhiễu. Các chỉ số đơn giản nhưng dễ giải thích có thể hỗ trợ quyết định xử lý thích nghi.

**Cách đã triển khai:**

Nhóm triển khai bước phân tích chất lượng ảnh gồm:

- `brightness`: giá trị trung bình của cường độ ảnh xám.
- `contrast`: độ lệch chuẩn của cường độ ảnh xám.
- `noise level`: độ lệch chuẩn của phần dư tần số cao sau khi so sánh ảnh với phiên bản đã làm mượt.

**Kết quả sau khi cải tiến:**

Pipeline có thêm thông tin để chọn preprocessing phù hợp hơn cho từng ảnh. Điều này giúp quá trình xử lý không còn phụ thuộc hoàn toàn vào một cấu hình cố định.

**Bài học rút ra:**

Phân tích chất lượng ảnh là bước quan trọng trước khi áp dụng thuật toán xử lý. Những chỉ số đơn giản như trung bình và độ lệch chuẩn vẫn có giá trị thực tế trong pipeline.

### Vấn đề 4: Cần chọn preprocessing phù hợp với từng ảnh

**Hiện tượng quan sát được:**

Khi áp dụng cùng một phương pháp preprocessing cho tất cả ảnh, một số ảnh được cải thiện nhưng một số ảnh khác lại xấu hơn. Ví dụ, ảnh tối cần tăng tương phản, ảnh nhiễu cần lọc, còn ảnh bình thường không nên xử lý quá mạnh.

**Nguyên nhân phân tích:**

Mỗi ảnh có đặc điểm khác nhau. Một phương pháp cố định không đủ linh hoạt để xử lý đồng thời các trường hợp tối, thiếu tương phản, nhiễu và ảnh đã có chất lượng tốt.

**Ý tưởng giải quyết:**

Dựa trên kết quả phân tích chất lượng ảnh, pipeline sẽ chọn preprocessing thích nghi:

- Ảnh tối hoặc tương phản thấp: dùng histogram equalization.
- Ảnh nhiễu: dùng median filter.
- Ảnh bình thường: dùng Gaussian filter để làm mượt nhẹ.

**Cách đã triển khai:**

Các thuật toán histogram equalization, median filter và Gaussian filter được triển khai thủ công bằng NumPy. Pipeline quyết định phương pháp xử lý dựa trên brightness, contrast và noise level.

**Kết quả sau khi cải tiến:**

Ảnh đầu vào được chuẩn bị phù hợp hơn trước khi segmentation. Kết quả mask ổn định hơn so với khi chỉ dùng một phương pháp tiền xử lý cho toàn bộ dataset.

**Bài học rút ra:**

Adaptive preprocessing giúp pipeline thực tế hơn vì ảnh trong dataset không đồng nhất. Tuy nhiên, các ngưỡng quyết định vẫn cần được kiểm thử và tinh chỉnh thêm.

### Vấn đề 5: Otsu grayscale ban đầu chưa ổn với banana và orange

**Hiện tượng quan sát được:**

Khi dùng Otsu thresholding trên ảnh xám, kết quả segmentation với banana và orange chưa ổn định. Do quả chuối và cam thường sáng, màu của chúng có thể gần với nền trắng khi chuyển sang grayscale. Có trường hợp Otsu chọn nhầm nền hoặc chỉ chọn được các vùng tối nhỏ.

**Nguyên nhân phân tích:**

Otsu hoạt động dựa trên histogram mức xám và giả định có thể tách hai nhóm cường độ tương đối rõ. Với quả sáng trên nền gần trắng, sự khác biệt trong grayscale không đủ mạnh. Thông tin màu bị mất khi chuyển RGB sang grayscale, làm giảm khả năng phân biệt quả với nền.

**Ý tưởng giải quyết:**

Bổ sung hướng segmentation có xét đến màu sắc thay vì chỉ dựa vào grayscale. Sau đó cải tiến thêm bằng cách so sánh màu pixel với màu nền ước lượng.

**Cách đã triển khai:**

Pipeline thêm color-aware segmentation và tiếp tục cải tiến bằng background color distance. Thay vì chỉ dùng ngưỡng xám, thuật toán xét mức độ khác biệt giữa màu pixel và màu nền gần trắng.

**Kết quả sau khi cải tiến:**

Kết quả với các loại quả sáng như banana và orange tốt hơn so với Otsu grayscale đơn thuần. Mask thu được phù hợp hơn cho bước trích xuất đặc trưng.

**Bài học rút ra:**

Otsu là thuật toán nền tảng dễ giải thích, nhưng không đủ cho mọi trường hợp. Với ảnh màu, giữ lại thông tin RGB có thể giúp segmentation thực tế hơn.

### Vấn đề 6: Ảnh rotated tạo viền đen ở biên ảnh

**Hiện tượng quan sát được:**

Nhiều ảnh trong dataset được tăng cường bằng phép xoay. Sau khi xoay, ảnh xuất hiện các vùng tam giác màu đen ở biên. Các vùng đen này đôi khi bị thuật toán chọn nhầm là foreground.

**Nguyên nhân phân tích:**

Khi ảnh được rotate, các vùng không có dữ liệu ảnh gốc thường được lấp bằng màu đen. Trong ảnh xám hoặc khi tính khác biệt so với nền trắng, vùng đen có độ khác biệt rất lớn nên dễ bị xem là đối tượng.

**Ý tưởng giải quyết:**

Loại bỏ các pixel rất tối trước hoặc trong quá trình tạo mask. Các pixel có cường độ gần đen ở biên ảnh được xem là artifact do augmentation, không phải vùng quả.

**Cách đã triển khai:**

Nhóm áp dụng ngưỡng black border threshold để loại các pixel rất tối khỏi mask ứng viên. Điều này giúp giảm ảnh hưởng của viền đen do rotation.

**Kết quả sau khi cải tiến:**

Các vùng tam giác đen ở biên ít bị chọn nhầm hơn. Mask tập trung tốt hơn vào vùng quả thay vì các artifact do xoay ảnh.

**Bài học rút ra:**

Dữ liệu đã augmentation có thể tạo ra artifact không tự nhiên. Khi thiết kế pipeline, cần kiểm tra trực tiếp ảnh thực tế thay vì chỉ giả định ảnh luôn sạch.

### Vấn đề 7: Non-white mask ban đầu chọn nhầm gần toàn bộ nền

**Hiện tượng quan sát được:**

Một cải tiến ban đầu thử dùng non-white background mask để tìm vùng không phải nền trắng. Tuy nhiên, kết quả mask trong nhiều ảnh bị quá lớn và chọn nhầm gần như toàn bộ nền.

**Nguyên nhân phân tích:**

Nền ảnh nhìn bằng mắt có vẻ trắng, nhưng trên thực tế nhiều pixel không hoàn toàn là màu trắng do bóng đổ, nén ảnh, nội suy sau rotation hoặc thay đổi ánh sáng. Nếu chỉ xem pixel không trắng là foreground, nhiều vùng nền cũng bị đưa vào mask.

**Ý tưởng giải quyết:**

Không nên dùng non-white mask trực tiếp như mask foreground. Thay vào đó, cần ước lượng màu nền và đo khoảng cách màu một cách linh hoạt hơn.

**Cách đã triển khai:**

Nhóm không tiếp tục dùng non-white mask như tiêu chí chính để chọn foreground. Cách tiếp cận được chuyển sang background color distance, kết hợp với loại bỏ viền đen và hậu xử lý mask.

**Kết quả sau khi cải tiến:**

Mask bớt bị phình ra toàn bộ nền. Việc chọn vùng quả dựa trên khác biệt màu so với nền hợp lý hơn so với kiểm tra pixel có phải trắng tuyệt đối hay không.

**Bài học rút ra:**

Trong ảnh thực tế, nền trắng hiếm khi là trắng tuyệt đối. Điều kiện quá cứng như “khác trắng thì là vật thể” dễ gây lỗi.

### Vấn đề 8: Largest non-border component có thể loại nhầm fruit

**Hiện tượng quan sát được:**

Một hướng xử lý từng thử là chọn largest non-border component, tức là bỏ các thành phần chạm biên ảnh rồi lấy thành phần lớn nhất còn lại. Tuy nhiên, một số ảnh quả, đặc biệt là banana hoặc ảnh đã bị dịch chuyển, có thể chạm hoặc nằm gần biên ảnh.

**Nguyên nhân phân tích:**

Giả định “vùng chạm biên là nền hoặc artifact” không luôn đúng. Trong dataset, quả có thể bị crop sát mép, bị translate sau augmentation hoặc có hình dạng dài như chuối nên dễ gần biên.

**Ý tưởng giải quyết:**

Thay vì loại toàn bộ component chạm biên, nên chọn component hợp lý dựa trên tỷ lệ diện tích. Cách này giảm nguy cơ loại nhầm quả thật.

**Cách đã triển khai:**

Nhóm thay thế tiêu chí largest non-border component bằng largest reasonable component selection. Component được đánh giá theo area ratio để giữ lại vùng có kích thước hợp lý, thay vì chỉ dựa vào việc có chạm biên hay không.

**Kết quả sau khi cải tiến:**

Một số ảnh quả nằm gần biên được giữ lại tốt hơn. Pipeline ít phụ thuộc vào giả định cứng về vị trí của quả trong ảnh.

**Bài học rút ra:**

Các quy tắc hậu xử lý mask cần phản ánh đặc điểm dữ liệu thực tế. Nếu quy tắc quá mạnh, thuật toán có thể loại bỏ đúng đối tượng cần tìm.

### Vấn đề 9: Cải tiến segmentation bằng background color distance

**Hiện tượng quan sát được:**

Sau khi thử Otsu grayscale và non-white mask, nhóm nhận thấy cần một cách ổn định hơn để tách quả màu đỏ, vàng, cam khỏi nền gần trắng. Các ảnh bright fruit trên white background là trường hợp khó nếu chỉ dùng mức xám.

**Nguyên nhân phân tích:**

Nền ảnh thường nằm ở vùng biên và có màu gần trắng, nhưng không đồng nhất tuyệt đối. Quả có màu khác nền theo không gian RGB, kể cả khi độ sáng grayscale tương đối gần nhau. Vì vậy, khoảng cách màu so với nền là tín hiệu hữu ích.

**Ý tưởng giải quyết:**

Ước lượng màu nền từ các pixel ở biên ảnh, bỏ qua các pixel viền đen do rotation. Sau đó, chọn các pixel có màu đủ khác so với màu nền làm ứng viên vùng quả.

**Cách đã triển khai:**

Pipeline ước lượng background color từ border pixels. Các pixel quá tối được bỏ qua khi tính màu nền để tránh ảnh hưởng của black border. Sau đó, thuật toán tính khoảng cách màu giữa từng pixel và màu nền ước lượng. Các pixel có khoảng cách đủ lớn được chọn làm fruit candidates, rồi tiếp tục áp dụng morphology và connected components để làm sạch mask.

**Kết quả sau khi cải tiến:**

Cách này hoạt động tốt hơn với các loại quả đỏ, vàng và cam trên nền gần trắng. Mask thường bao phủ vùng quả hợp lý hơn, đặc biệt trong các trường hợp Otsu grayscale không tách được rõ.

**Bài học rút ra:**

Khi nền có thể ước lượng được, background modeling đơn giản vẫn đem lại hiệu quả thực tế. Không phải lúc nào cần dùng mô hình phức tạp; quan trọng là chọn tín hiệu phù hợp với dữ liệu.

### Vấn đề 10: Nền có texture hoặc fruit có vùng mốc/trắng gây khó khăn

**Hiện tượng quan sát được:**

Một số ảnh rotten orange có nền có texture hoặc vùng mốc/trắng trên quả. Những vùng này có thể làm mask xuất hiện lỗ, hoặc tạo thêm artifact nền bị chọn nhầm.

**Nguyên nhân phân tích:**

Vùng mốc/trắng trên quả có màu gần với nền, nên khoảng cách màu so với background nhỏ hơn các vùng quả còn lại. Ngược lại, nền có texture hoặc bóng đổ có thể khác màu nền trung bình, khiến thuật toán nhầm là foreground.

**Ý tưởng giải quyết:**

Chấp nhận rằng segmentation hiện tại chỉ cần đủ tốt để trích xuất đặc trưng, không cần đạt pixel-perfect mask. Tiếp tục dùng morphology và connected components để giảm lỗi lớn, đồng thời ghi nhận đây là hạn chế cần cải tiến.

**Cách đã triển khai:**

Pipeline sử dụng mask sau background color distance kết hợp morphology cleanup và lựa chọn component hợp lý. Các trường hợp khó được ghi nhận để đưa vào phần hạn chế và hướng cải tiến.

**Kết quả sau khi cải tiến:**

Kết quả hiện tại chấp nhận được cho mục tiêu trích xuất đặc trưng tổng quát. Tuy nhiên, một số ảnh khó vẫn có mask chưa hoàn hảo, đặc biệt ở vùng mốc trắng hoặc nền có texture.

**Bài học rút ra:**

Segmentation trong dự án này là công cụ hỗ trợ feature extraction, không phải mục tiêu cuối cùng. Cần trình bày trung thực rằng mask còn hạn chế và chưa có ground-truth để đánh giá pixel-level.

## 4. Bảng tóm tắt các cải tiến

| Giai đoạn | Vấn đề | Giải pháp | Kết quả |
|---|---|---|---|
| Dataset sampling | Dataset gốc quá lớn, khó kiểm thử nhanh và không phù hợp đưa lên GitHub | Tạo tập `data/sample` cân bằng từ `data/raw` với random seed `42` | Dữ liệu nhỏ hơn, dễ kiểm thử, vẫn giữ tính cân bằng giữa các lớp |
| Quality analysis | Ảnh có độ sáng, tương phản và nhiễu khác nhau | Tính brightness, contrast và noise level từ ảnh xám | Có cơ sở để chọn preprocessing thích nghi |
| Adaptive preprocessing | Một phương pháp cố định không phù hợp với mọi ảnh | Chọn histogram equalization, median filter hoặc Gaussian filter tùy chất lượng ảnh | Ảnh đầu vào ổn định hơn trước segmentation |
| Otsu thresholding | Otsu grayscale chưa tốt với quả sáng trên nền trắng | Bổ sung hướng color-aware segmentation | Giảm phụ thuộc vào histogram mức xám |
| Black border removal | Ảnh rotate tạo viền đen bị chọn nhầm là foreground | Loại pixel rất tối bằng black border threshold | Giảm artifact ở biên ảnh |
| Background color distance | Nền trắng không hoàn toàn trắng, quả sáng khó tách bằng grayscale | Ước lượng màu nền từ border pixels và chọn pixel khác nền | Tách tốt hơn các quả đỏ, vàng, cam trên nền gần trắng |
| Reasonable component selection | Loại component chạm biên có thể loại nhầm quả | Chọn component hợp lý theo area ratio | Giữ lại tốt hơn các quả gần biên hoặc bị translate |
| Morphology cleanup | Mask có lỗ nhỏ hoặc nhiễu rời rạc | Áp dụng các bước morphology và connected components | Mask sạch hơn, phù hợp hơn cho feature extraction |

## 5. Kết quả kiểm thử hiện tại

Các unit tests hiện tại đã chạy pass. Ngoài kiểm thử bằng test tự động, nhóm cũng đã kiểm tra trên ảnh thật lấy mẫu từ cả 6 lớp trong dataset:

- `freshapples`
- `freshbanana`
- `freshoranges`
- `rottenapples`
- `rottenbanana`
- `rottenoranges`

Việc kiểm thử được thực hiện trên nhiều ảnh ngẫu nhiên từ mỗi lớp để quan sát chất lượng mask trong các trường hợp khác nhau. Nhìn chung, segmentation hiện tại đủ tốt để làm bước ước lượng vùng quả phục vụ trích xuất đặc trưng. Tuy nhiên, một số ảnh khó vẫn có mask chưa hoàn hảo, nhất là khi nền có texture, quả chạm biên, có bóng đổ hoặc vùng mốc/trắng gần màu nền.

## 6. Hạn chế hiện tại

- Segmentation hiện tại không phải pixel-perfect và không nên được trình bày như ground-truth annotation.
- Dự án chưa có ground truth masks để đánh giá định lượng ở mức pixel.
- Một số mask có thể bao gồm lá, cuống, bóng đổ hoặc nhiều quả trong cùng ảnh.
- Một số vùng hư hỏng màu trắng hoặc vùng mốc trên quả có thể bị bỏ sót một phần vì gần màu nền.
- Các ngưỡng hiện tại mang tính thực nghiệm và cần parameter sweep để chọn giá trị tốt hơn.
- Mục tiêu chính của segmentation là ước lượng vùng quả để trích xuất đặc trưng, không phải giải quyết hoàn chỉnh bài toán phân đoạn ảnh.

## 7. Hướng cải tiến tiếp theo

Các bước cải tiến tiếp theo có thể tập trung vào trích xuất đặc trưng, huấn luyện mô hình và hoàn thiện ứng dụng:

- Trích xuất đặc trưng hình dạng: area, perimeter, circularity, bounding box và aspect ratio.
- Trích xuất đặc trưng màu sắc và chất lượng ảnh: color histogram, brightness, contrast và noise level.
- Bổ sung các đặc trưng liên quan đến khuyết tật hoặc vùng hư hỏng của quả.
- Xuất toàn bộ đặc trưng ra file CSV để phục vụ huấn luyện machine learning.
- Huấn luyện và so sánh các mô hình Random Forest, KNN và SVM.
- Xây dựng luật đánh giá export suitability dựa trên kết quả phân loại và đặc trưng chất lượng.
- Xây dựng GUI cho phép người dùng chọn ảnh, xem kết quả phân tích và đọc giải thích ngắn gọn.
- Thực hiện parameter sweep để chọn ngưỡng tốt hơn và có thêm số liệu đưa vào báo cáo.

## 8. Ghi chú cho báo cáo và bảo vệ

- Nhóm đã tự triển khai các thuật toán xử lý ảnh lõi bằng NumPy, chỉ dùng PIL cho đọc và lưu ảnh.
- Pipeline được cải tiến dần thông qua kiểm thử trên ảnh thật từ nhiều lớp trái cây.
- Otsu thresholding đơn thuần không đủ ổn định cho tất cả loại quả, đặc biệt với quả sáng trên nền trắng.
- Background color distance được bổ sung để xử lý tốt hơn các trường hợp quả đỏ, vàng và cam trên nền gần trắng.
- Segmentation trong dự án được dùng như mask thực tế để trích xuất đặc trưng, không phải annotation hoàn hảo ở mức pixel.

## Khó khăn: Một số ảnh tươi bị dự đoán nhầm thành hư

**Hiện tượng:**

Một ảnh `fresh apple` bị dự đoán thành `rotten` trong quá trình test 6 class.

**Nguyên nhân có thể:**

Giọt nước, lá, vùng sáng mạnh, texture phức tạp hoặc artifact có thể bị defect map hiểu nhầm là vùng lỗi.

**Cách xử lý hiện tại:**

Giữ logic theo hướng an toàn cho bài toán xuất khẩu. Nếu hệ thống nghi ngờ chất lượng, kết luận `Not Suitable` hoặc `Need Recheck` là chấp nhận được.

**Lý do không sửa thuật toán ngay:**

Pipeline tổng thể đã ổn, test tự động pass, manual test đa số đúng. Việc chỉnh threshold sớm có thể làm giảm khả năng phát hiện trái cây hư thật.

**Hướng cải thiện:**

Tinh chỉnh defect map, lọc highlight/viền tốt hơn, tăng dữ liệu train, đánh giá thêm nhiều ảnh, và cân nhắc threshold riêng theo từng loại trái cây.

## Buoc 10: Quyet dinh phan hang thi truong cuoi cung

Du an bo sung lop phan hang thi truong cuoi cung de bien ket qua ky thuat thanh quyet dinh gan voi ung dung thuc te. Sau khi he thong da du doan loai trai cay, chat luong `fresh` hoac `rotten`, ty le khuyet tat va do tin cay cua mat na phan doan, nguoi dung van can mot ket luan ro rang hon de sap xep trai cay: xuat khau, noi dia, hoac loai bo.

Cac nhan `Suitable`, `Need Recheck`, va `Not Suitable` huu ich cho buoc danh gia kha nang xuat khau, nhung chua du de dap ung muc tieu cuoi cung cua ung dung. Trong thuc te, trai cay khong chi duoc danh gia la co phu hop xuat khau hay khong, ma con can duoc dua vao kenh phan phoi phu hop. Vi du, trai cay con tuoi nhung co mot so khuyet tat nho co the khong dat xuat khau, nhung van co the ban o thi truong noi dia.

Du an anh xa ket qua ky thuat sang quyet dinh sap xep thuc te bang cac luat don gian. Neu mo hinh du doan trai cay bi hong, he thong gan `Reject`. Neu trai cay con tuoi va `defect_ratio` thap, he thong gan `Export Grade`. Neu trai cay con tuoi nhung `defect_ratio` o muc trung binh hoac phan doan hoi dang nghi, he thong gan `Domestic Grade`. Neu mat na phan doan qua nho hoac qua lon, he thong co the gan `Domestic Grade` hoac `Reject` tuy muc do nghiem trong.

Cach phan hang dua tren luat phu hop voi du an xu ly anh truyen thong vi de giai thich, de bao ve trong bao cao, va khong phu thuoc vao mang hoc sau. Lop quyet dinh nay cung khong lam thay doi cac thuat toan xu ly anh va khong can huan luyen lai mo hinh, nen phu hop voi muc tieu giu he thong don gian, ro rang va co tinh giao duc.

Han che hien tai la cac nguong nhu `defect_ratio` va `mask_area_ratio` van mang tinh kinh nghiem. Ket qua co the bi anh huong boi anh chup qua toi, nen phuc tap, bong do, vat the khac trong anh, hoac phan doan chua chinh xac. Trong tuong lai, cac nguong nay nen duoc hieu chinh bang tap du lieu lon hon va danh gia boi nguoi co kinh nghiem trong phan loai nong san.
