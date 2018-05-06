#include <iostream>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "imgDetectFunctions.h"
#include "include/inputModel.h"


using namespace cv;
CvMemStorage            *storage;


void openWithCam(CvMemStorage *aStorage);
void openWithImg();

int main(int argc, char *argv[]) {

   openWithCam(storage);
//    openWithImg();

    cvReleaseMemStorage( &storage );

    return 0;

}
