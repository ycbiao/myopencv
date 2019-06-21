package com.example.opencvtest.fun;

import com.example.opencvtest.common.Learn;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.StatModel;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.FileFilter;
import java.util.*;

import static com.sun.org.apache.xalan.internal.utils.SecuritySupport.getResourceAsStream;
import static java.lang.String.format;
import static org.opencv.core.CvType.CV_32F;

public class FaceTest {

    public static Mat faceDetectionTest(){

        String facetestfile = "";
        try {
            facetestfile = ResourceUtils.getFile("classpath:pic/facetest.png").getAbsolutePath();
        }catch (Exception e){
            e.printStackTrace();
        }
        Mat srcImg = Imgcodecs.imread(facetestfile,1);
        List<Rect> facess = getFaceRect(srcImg);
        for(int i =0; i < facess.size();i++ ){
            Imgproc.rectangle(srcImg, facess.get(i), new Scalar(0, 0, 255), 1, 8, 0);
        }
        return srcImg;
    }

    /**
     * 获取人脸数据rect
     */
    public static List<Rect> getFaceRect(Mat srcImg){

        try {
            //        String cascadeFilePath = "G:\\chromedownlaods\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
            String cascadeFilePath = ResourceUtils.getFile("classpath:xml/haarcascade_frontalface_alt.xml").getAbsolutePath();
            CascadeClassifier cascadeClassifier =  new CascadeClassifier();
            if(!cascadeClassifier.load(cascadeFilePath)){
                System.out.print("could not load the haar data...\\n");
            }
//        Mat srcImg = Imgcodecs.imread("C:\\Users\\ycb\\Desktop\\base64\\facetest.jpg",1);
            Mat srcImgGray = new Mat();
            Imgproc.cvtColor(srcImg,srcImgGray,Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(srcImgGray,srcImgGray);

            MatOfRect faces = new MatOfRect();
            cascadeClassifier.detectMultiScale(srcImgGray,faces);

            List<Rect> facess = faces.toList();
            return facess;
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }

    /**
     * 人脸识别
     */
    public static void faceRecognitionTest(){
        String saveMyPicPath ="";
        try {
             saveMyPicPath = ResourceUtils.getFile("classpath:attfaces/").getAbsolutePath();
        }catch (Exception e){
            e.printStackTrace();
        }

        List<Mat> matList = new ArrayList<>();

        for (int i = 1;i <=10;i++){
            String filename = String.format("C:\\Users\\ycb\\Desktop\\base64\\zhang%d.jpg", i);
            Mat srcImg = Imgcodecs.imread(filename,1);
            List<Rect> faceRects = getFaceRect(srcImg);

            for(int j = 0; j < faceRects.size();j++ ){
                //切割出人脸
                Mat myFace = new Mat(srcImg,faceRects.get(j));
                if(myFace.cols() > 100){
                    Imgproc.resize(myFace, myFace, new  Size(92, 112));
                }
//                myFace.convertTo(myFace,Imgproc.COLOR_BGR2GRAY);
                Mat srcImgGray = new Mat();
                Imgproc.cvtColor(myFace,srcImgGray,Imgproc.COLOR_BGR2GRAY, CV_32F);
//                HighGui.imshow("H", srcImgGray);
//                HighGui.waitKey(10);
                String filepath = "C:\\Users\\ycb\\Desktop\\base64" + String.format("\\%d.jpg",i) ;
                System.out.print("save my pic is " + Imgcodecs.imwrite(filepath,srcImgGray) + "\n");
                matList.add(srcImgGray);
            }
        }

        readFace(saveMyPicPath,matList);
    }

    /**
     * 读取attfaces下样本共40，还有自己的人脸数据，放在41位置
     * @param filePath
     */
    public static void  readFace(String filePath,List<Mat> myMatList){
        List<String> files = new ArrayList<String>();
        File file = new File(filePath);
        File[] tempList = file.listFiles();

        List<File> fileList = Arrays.asList(tempList);
        Collections.sort(fileList, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                //升序取文件夹
                String o1name = o1.getAbsolutePath();
                o1name = o1name.substring(o1name.lastIndexOf("s") + 1,o1name.length());
                String o2name =  o2.getAbsolutePath();
                o2name = o2name.substring(o2name.lastIndexOf("s") + 1,o2name.length());

                if(Integer.valueOf(o1name) > Integer.valueOf(o2name))
                    return 1;
                else
                    return -1;
            }
        });

        Mat trainDataMat =  null;
        Mat labelsMat = new Mat(fileList.size() + 1,1,CvType.CV_32SC1);
        for (int i = 0; i < fileList.size() + 1; i++) {
            if(i == fileList.size()){
                //自己的人脸数据
                int l = 0;
                for (Mat mat : myMatList){
                    for (int j = 0 ; j < mat.rows() ; j++){
                        for (int k = 0 ; k < mat.cols() ; k++){
                            double[] doubles = mat.get(j,k);
                            trainDataMat.put(i,l,doubles[0]);
                            l ++ ;
                        }
                    }
                }
                labelsMat.put(i,0,i);
                break;
            }

            File file1 = fileList.get(i);
            if(file1.isDirectory()){
                //样本的脸数据
                File[] files1 = file1.listFiles();
                for (File file2 : files1){
                    Mat mat = Imgcodecs.imread(file2.getAbsolutePath());
                    int count = mat.rows() * mat.cols();
                    if(trainDataMat == null){
                        trainDataMat =  new Mat(fileList.size() + 1,count,CvType.CV_32FC1);
                    }

                    int l = 0;
                    for (int j = 0 ; j < mat.rows() ; j++){
                        for (int k = 0 ; k < mat.cols() ; k++){
                            double[] doubles = mat.get(j,k);
                            trainDataMat.put(i,l,doubles[0]);
                            l ++ ;
                        }
                    }
                }
                labelsMat.put(i,0,i);
            }
        }

        //训练
        StatModel statModel;
        statModel= Learn.svm(trainDataMat,labelsMat);

        //测试
        Mat matTest = Imgcodecs.imread("C:\\Users\\ycb\\Desktop\\base64\\test.jpg");
        List<Rect> rectList = getFaceRect(matTest);

        for(int j = 0; j < rectList.size();j++ ){
            //切割出人脸
            Mat myFace = new Mat(matTest,rectList.get(j));
//            if(myFace.cols() > 100){
                Imgproc.resize(myFace, myFace, new  Size(92, 112));
//            }
//            Mat srcImgGray = new Mat();
            Imgproc.cvtColor(myFace,matTest,Imgproc.COLOR_BGR2GRAY, CV_32F);
                HighGui.imshow("H", matTest);
                HighGui.waitKey(10);
        }

//        HighGui.imshow("H", matTest);
//        HighGui.waitKey(10);
        Mat sampleMat =  new Mat(1, matTest.rows() * matTest.cols(), CV_32F);
        int m = 0;
        for (int j = 0 ; j < matTest.rows() ; j++){
            for (int k = 0 ; k < matTest.cols() ; k++){
                sampleMat.put(0,m,matTest.get(j,k));
                m ++ ;
            }
        }
        System.out.print("测试人脸在样本位置 = " + (int)(statModel.predict(sampleMat) + 1));
    }
}
