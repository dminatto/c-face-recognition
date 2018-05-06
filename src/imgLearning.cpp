#include "imgLearning.h"

#include "opencv/cvaux.h"
#include "opencv/ml.h"
#include "opencv/cv.h"

#include <iostream>

using namespace std;
using namespace cv;
using namespace ml;


imgLearning::imgLearning()
{

}

imgLearning::~imgLearning()
{
    //dtor
}

void addlogNN(Mat aSample, Mat aResult);
float evaluateLearning(cv::Mat& predicted, cv::Mat& actual) {
    assert(predicted.rows == actual.rows);
    int t = 0;
    int f = 0;
    for(int i = 0; i < actual.rows; i++) {
        float p = predicted.at<float>(i,0);
        float a = actual.at<float>(i,0);
        if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
            t++;
        } else {
            f++;
        }
    }
    return (t * 1.0) / (t + f);
}

void detectAndLearning(Mat matInputLayers, Mat matTrainLabels){

Mat classificationResult;
Mat predicted(matInputLayers.rows, 1, CV_32F);

CvMat test_sample;
int number_testing_samples = 6367;
//int number_testing_samples = 10;


cv::Ptr<cv::ml::ANN_MLP> mlp;

mlp = cv::ml::ANN_MLP::create();

cv::Mat layers = cv::Mat(4, 1, CV_32SC1);
layers.row(0) = cv::Scalar(4);
layers.row(1) = cv::Scalar(88);
layers.row(2) = cv::Scalar(30);
layers.row(3) = cv::Scalar(1);

CvTermCriteria criteria;

criteria.max_iter = 200;

criteria.epsilon = 0.0000001f;

criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

mlp->setLayerSizes(layers);

mlp->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);

mlp->setTermCriteria(criteria);

mlp->setTrainMethod(ml::ANN_MLP::BACKPROP);

mlp->setBackpropMomentumScale(0.05f);

mlp->setBackpropWeightScale(0.05f);

//treinamento
cout << "inicinado o trenamento" << endl;
Ptr<ml::TrainData> trainData = ml::TrainData::create(matInputLayers, ml::ROW_SAMPLE, matTrainLabels);

mlp->train(trainData);

for (int i = 0; i < matInputLayers.rows; i++){

    Mat sample = Mat(1, matInputLayers.cols, CV_32F, matInputLayers.at<int>(i));
    Mat result;
    mlp->predict(sample, result);
    cout << sample << " -> " << result << matTrainLabels.row(i) << endl;
    predicted.at<float>(i,0) = result.at<float>(0,0);
    addlogNN(sample, result);
    }


 cout << "Accuracy_{MLP} = " << evaluateLearning(predicted, matInputLayers) << endl;


}
