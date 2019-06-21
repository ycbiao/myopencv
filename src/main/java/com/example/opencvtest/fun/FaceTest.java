package com.example.opencvtest.fun;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.util.List;

public class FaceTest {

    public static Mat run(){
//        String cascadeFilePath = "G:\\chromedownlaods\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
        String cascadeFilePath = "";
        String facetestfile = "";
        try {
            cascadeFilePath = ResourceUtils.getFile("classpath:xml/haarcascade_frontalface_alt.xml").getAbsolutePath();
            facetestfile = ResourceUtils.getFile("classpath:pic/facetest.png").getAbsolutePath();
        }catch (Exception e){
            e.printStackTrace();
        }
        CascadeClassifier cascadeClassifier =  new CascadeClassifier();
        if(!cascadeClassifier.load(cascadeFilePath)){
            System.out.print("could not load the haar data...\\n");
        }
//        Mat srcImg = Imgcodecs.imread("C:\\Users\\ycb\\Desktop\\base64\\facetest.jpg",1);
        Mat srcImg = Imgcodecs.imread(facetestfile,1);
        Mat srcImgGray = new Mat();
        Imgproc.cvtColor(srcImg,srcImgGray,Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(srcImgGray,srcImgGray);

        MatOfRect faces = new MatOfRect();
        cascadeClassifier.detectMultiScale(srcImgGray,faces);

        List<Rect> facess = faces.toList();
        for(int i =0; i < facess.size();i++ ){
            Imgproc.rectangle(srcImg, facess.get(i), new Scalar(0, 0, 255), 1, 8, 0);
        }

        return srcImg;
    }
}
