#ifndef INPUTMODEL_H
#define INPUTMODEL_H
#include "opencv2/highgui/highgui.hpp"
#include <string>

using namespace std;
using namespace cv;

class inputModel
{
    public:
        inputModel();
        void createlog(string sName);
        void addlog(double aiImagem, double asCaracteristica, double aiTotal, double aiCorrente);
        virtual ~inputModel();
        void openWithCam(CvMemStorage *aStorage);
        void openWithImg();
        void storeAttainment(float x, float y, float h, float w, float r);
    protected:
    private:
};

#endif // INPUTMODEL_H


