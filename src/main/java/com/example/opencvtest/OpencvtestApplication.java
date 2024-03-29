package com.example.opencvtest;

import com.example.opencvtest.fun.FaceTest;
import com.example.opencvtest.fun.SvmTest;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;

@SpringBootApplication
public class OpencvtestApplication {
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    public static void main(String[] args) {
        SpringApplicationBuilder builder = new SpringApplicationBuilder(OpencvtestApplication.class);
        builder.headless(false).web(WebApplicationType.NONE).run(args);

//        Mat m = Imgcodecs.imread("C:\\Users\\ycb\\Desktop\\base64\\1.jpg",1);
//
//        if (m.empty()){
//            System.out.println("图片不存在");
//            return;
//        }
//
//        Mat gray = new Mat();
//        Imgproc.cvtColor(m, gray, Imgproc.COLOR_BGR2GRAY);
//        Mat dst = new Mat(m.rows(),m.cols(),m.type());//此时的 dst 是8u1c
//        Imgproc.Canny(gray, dst, 130, 250);
//
//        System.out.println("OpenCV Mat: " + dst);
//        HighGui.imshow("H",dst);
//        HighGui.waitKey(1000);
//        //        System.out.println("OpenCV Mat data:\n" + m.dump());


//        HighGui.imshow("H", FaceTest.faceDetectionTest());
//        HighGui.waitKey(1000);
        FaceTest.faceRecognitionTest();
    }

}
