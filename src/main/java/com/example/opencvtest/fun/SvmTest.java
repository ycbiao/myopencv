package com.example.opencvtest.fun;

import com.example.opencvtest.common.Learn;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.*;

import static org.opencv.core.CvType.*;
import static org.opencv.core.CvType.CV_32F;

public class SvmTest {
    public static Mat test() {
        int width = 512, height = 512;                        //512*512 的正方形区域
        Mat image = new Mat(width, height, CV_8UC3);

//        int labels[] = { 1, 1, 1, 1, 1, 1 ,0, 0, 0, 0};			//8 个结果
//        Mat labelsMat = new Mat(labels.length, 1, CV_32S);
//        labelsMat.put(0,0,labels);

        float labels[][] = {
                {1, 1},                //8 样本点（和结果对应）
                {1, 1},
                {1, 1},
                {1, 1},
                {0, 0},
                {0, 0},
                {0, 0},
                {0, 0},
                {0, 0},
                {0, 0}
        };

        Mat labelsMat = new Mat(labels.length, labels[0].length, CV_32F);
        for (int k = 0;k < labels.length ;k++){
            for (int l = 0 ;l < 2; l++){
                labelsMat.put(k,l,labels[k][l]);
            }

        }

        float trainingData[][] = {
                {10, 10},				//8 样本点（和结果对应）
                {10, 50},
                {40,30},
                {70, 60},
                {100, 100},
                {120, 120},
                {501, 255},
                {500, 501},
                {300,300},
                {60, 500}};


        Mat trainingDataMat =  new Mat(trainingData.length, trainingData[0].length, CV_32F);
        for (int i = 0;i < trainingData.length ;i++){
            for (int j = 0 ;j < 2; j++){
                trainingDataMat.put(i,j,trainingData[i][j]);
            }

        }

        StatModel statModel;
//         statModel= Learn.svm(trainingDataMat,labelsMat);
//         statModel= Learn.boost(td);

//        Mat labelsMat1 = new Mat();
//        labelsMat.convertTo(labelsMat1,CV_32F);
//         statModel= Learn.logisticRegression(trainingDataMat,labelsMat1);
//        statModel = Learn.myNormalBayes(trainingDataMat,labelsMat);
//        statModel = Learn.myRTrees(trainingDataMat,labelsMat);

        statModel = Learn.myAnn(trainingDataMat,labelsMat);

        Mat sampleMat =  new Mat(1, 2, CV_32F);

        float response;
        for (int i = 0; i < image.rows(); ++i)
            for (int j = 0; j < image.cols(); ++j)
            {
                sampleMat.put(0,0,i);
                sampleMat.put(0,1,j);
                response = statModel.predict(sampleMat);

                if (response == 1)		//1画绿色
                    image.put(i,j, 0, 255, 0);
//                image.put(i,j, 255, 0, 0);
                else if (response == 0)	//0画蓝色
                    image.put(i,j, 255, 0, 0);
            }


        HighGui.imshow("H",image);
        HighGui.waitKey(1000);

        // 标出样本点的位置
        int thickness = -1;
        int lineType = 8;
        float x, y;
        Scalar s;
        for (int i = 0; i < labels.length; i++) {
            if (labels[i][0] == 1) {
                s = new Scalar(255, 0, 255);
            } else {
                s = new Scalar(255, 255, 0);
            }
            x = trainingData[i][0];
            y = trainingData[i][1];
            Imgproc.circle(image, new Point(x, y), 5, s, thickness, lineType);
        }
        return  image;
    }
}
