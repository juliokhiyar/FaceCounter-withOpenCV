#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) {
        std::cerr << "Error opening camera." << std::endl;
        return -1;
    }

    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cv::samples::findFile("haarcascades/haarcascade_frontalface_alt.xml"))) {
        std::cerr << "Error loading face cascade." << std::endl;
        return -1;
    }

    int faceCount = 0;  
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error reading frame." << std::endl;
            break;
        }
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 4);

        for (const auto& rect : faces) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        }

        faceCount = faces.size();
        cv::putText(frame, "Jumlah Wajah: " + std::to_string(faceCount), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Face Detection", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    cap.release();

    return 0;
}
