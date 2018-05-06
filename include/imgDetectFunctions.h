#ifndef IMGDETECTFUNCTIONS_H

#define IMGDETECTFUNCTIONS_H

#include "imgLearning.h"
#include "inputModel.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <string>

using namespace std;
using namespace cv;

extern string IS_FACE;
extern string NOT_FACE;

extern char *FACE_CASCADE;
extern char *EYE_CASCADE;
extern char* CASCADE_MOUTH;

class imgDetectFunctions {

    public:
        imgDetectFunctions();
        virtual ~imgDetectFunctions();
        void detectAndDisplay(IplImage* amframe, int aimg_no, string atype);
    private:
        void detectEyes(IplImage* iFature, CvRect* arPos, CvMemStorage* aStorage, int aimg_no);
        void detectMouth(IplImage* iFature, CvRect* arPos, CvMemStorage* aStorage, int aimg_no);
        IplImage* transformImgs(IplImage* amframe);

};

#endif // IMGDETECTFUNCTIONS_H
