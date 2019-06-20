package com.example.opencvtest.common;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.*;

/**
 * 机器学习方法汇总
 */
public class Learn {
    /**
     * // 支持向量机
     * @param
     * @return
     */
    public static StatModel svm(Mat trainingData, Mat labels){
        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        SVM svm = SVM.create();
//        // 配置SVM训练器参数
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 1000, 0);
        svm.setTermCriteria(criteria);// 指定
        svm.setKernel(SVM.LINEAR);// 使用预先定义的内核初始化
        svm.setType(SVM.NU_SVC); // SVM的类型,默认是：SVM.C_SVC
        svm.setGamma(0.5);// 核函数的参数
        svm.setNu(0.5);// SVM优化问题参数
        svm.setC(1);// SVM优化问题的参数C
//        svm.save("./Result/f.xml");//存储模型
        System.out.println("Training result: " + svm.train(td));
        return svm;
    }

    /**
     * // boost
     * @param td
     * @return
     */
    public static StatModel boost(TrainData td){
        Boost boost = Boost.create();
        // boost.setBoostType(Boost.DISCRETE);
        boost.setBoostType(Boost.REAL);
        boost.setWeakCount(2);
        boost.setWeightTrimRate(0.95);
        boost.setMaxDepth(2);
        boost.setUseSurrogates(false);
        boost.setPriors(new Mat());
        System.out.println("Training result: " + boost.train(td));
        return boost;
    }

    /**
     * 逻辑回归
     * @param
     * @return
     */
    public static StatModel logisticRegression(Mat trainingData, Mat labels){
        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        LogisticRegression lr = LogisticRegression.create();
        boolean success = lr.train(td);
        System.out.println("LogisticRegression training result: " + success);
        return lr;
    }

    /**
     * 贝叶斯
     * @param trainingData
     * @param labels
     * @return
     */
    public static StatModel myNormalBayes(Mat trainingData, Mat labels){
        NormalBayesClassifier nb = NormalBayesClassifier.create();

        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        boolean success = nb.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        System.out.println("NormalBayes training result: " + success);
        return nb;
    }

    /**
     * 随机森林
     * @param trainingData
     * @param labels
     * @return
     */
    public static StatModel myRTrees(Mat trainingData, Mat labels){
        RTrees rtrees = RTrees.create();
        rtrees.setMaxDepth(4);
        rtrees.setMinSampleCount(2);
        rtrees.setRegressionAccuracy(0.f);
        rtrees.setUseSurrogates(false);
        rtrees.setMaxCategories(16);
        rtrees.setPriors(new Mat());
        rtrees.setCalculateVarImportance(false);
        rtrees.setActiveVarCount(1);
        rtrees.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 500, 0));
        TrainData tData = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        boolean success = rtrees.train(tData.getSamples(), Ml.ROW_SAMPLE, tData.getResponses());
        System.out.println("Rtrees training result: " + success);
        return rtrees;
    }

    /**
     * 人工神经网络
     * @param trainingData
     * @param labels
     * @return
     */
    public static StatModel myAnn(Mat trainingData, Mat labels){
        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        Mat layerSizes = new Mat(trainingData.rows(), trainingData.cols(), CvType.CV_32FC1);
        // 含有两个隐含层的网络结构，输入、输出层各两个节点，每个隐含层含两个节点
//        layerSizes.put(0, 0, new float[] { 2 });
        ANN_MLP ann = ANN_MLP.create();
        ann.setLayerSizes(layerSizes);
        ann.setTrainMethod(ANN_MLP.BACKPROP);
        ann.setBackpropWeightScale(0.1);
        ann.setBackpropMomentumScale(0.1);
        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM, 1, 1);
        ann.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 300, 0.0));
        boolean success = ann.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        System.out.println("Ann training result: " + success);
        return ann;
    }
}
