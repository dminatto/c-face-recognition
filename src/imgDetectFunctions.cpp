#include "imgDetectFunctions.h"

#include "opencv/cv.h"
#include "opencv/cvaux.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <string>
#include <iostream>

using namespace std;

string IS_FACE = "Y";
string NOT_FACE = "N";

char *CASCADE_MOUTH = "haarcascade_mouth.xml";
char *FACE_CASCADE  ="haarcascade_frontalface_alt.xml";
char *EYE_CASCADE   = "haarcascade_eyes.xml.xml";


void storeAttainment(float x, float y, float h, float w, float r);
void addlog(int aiImagem, string asCaracteristica, int aiTotal, int aiCorrente);
imgDetectFunctions::imgDetectFunctions(){



 }

imgDetectFunctions::~imgDetectFunctions(){


}


void detectMouth(IplImage* iFature, CvRect* arPos, CvMemStorage* aStorage, int aimg_no){

    CvSeq                   *csMouth;
    CvHaarClassifierCascade *lcMouth;

    lcMouth = (CvHaarClassifierCascade*)cvLoad(CASCADE_MOUTH, 0, 0, 0);

    Rect rMouth(arPos->x, (arPos->y + (arPos->height*0.68)), arPos->width, (arPos->height/3));

    cvSetImageROI(iFature, rMouth);

    csMouth = cvHaarDetectObjects(iFature,
                                  lcMouth,
                                  aStorage,
                                  1.10, 1, 0,
                                  cvSize(25, 15));

    for( int i = 0; i < (csMouth ? csMouth->total : 0); i++ )
    {
        CvRect *crMouth = (CvRect*)cvGetSeqElem(csMouth, i);

        cvRectangle(iFature,
                    cvPoint(crMouth->x, crMouth->y),
                    cvPoint(crMouth->x + crMouth->width, crMouth->y + crMouth->height),
                    CV_RGB(255,255, 255),
                    2, 8, 0
                   );

        addlog(aimg_no, "MOUTH", csMouth->total, i);
    }

    cvReleaseHaarClassifierCascade( &lcMouth );


}

void detectEyes(IplImage* iFature, CvRect* arPos, CvMemStorage* aStorage, int aimg_no){

    CvSeq                   *csEyes;
    CvHaarClassifierCascade *lcEyes;

    lcEyes = (CvHaarClassifierCascade*)cvLoad(EYE_CASCADE, 0, 0, 0);

    Rect rEyes(arPos->x,arPos->y + (arPos->height/5.5), arPos->width, arPos->height/3.0);

    cvSetImageROI(iFature, rEyes);

    csEyes = cvHaarDetectObjects(iFature,
                                 lcEyes,
                                 aStorage,
                                 1.05, 3, 0,
                                 cvSize(25, 15));

    for( int i = 0; i < (csEyes ? csEyes->total : 0); i++ )
    {
        CvRect *crEye = (CvRect*)cvGetSeqElem(csEyes, i);

        cvRectangle(iFature,
                    cvPoint(crEye->x, crEye->y),
                    cvPoint(crEye->x + crEye->width, crEye->y + crEye->height),
                    CV_RGB(0, 0, 255),
                    2, 8, 0 );

        addlog(aimg_no, "EYE", csEyes->total, 0);
    }

    cvReleaseHaarClassifierCascade( &lcEyes );

}

/** função principal de detectação
encontra a face e direciona para as outras funções de detecção*/

void detectAndDisplay( IplImage* amFrame, int aimg_no, string type) {

    char image[100];
    int itype_awnser = 0;

    CvRect                  *r;
    CvSeq                   *csFace ;
    CvMemStorage            *storage = cvCreateMemStorage(0);
    CvHaarClassifierCascade *lcFace;

    cvClearMemStorage( storage );

    lcFace  = (CvHaarClassifierCascade*)cvLoad(FACE_CASCADE, 0, 0, 0);

   if( lcFace ){
    csFace = cvHaarDetectObjects(amFrame, lcFace, storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, cvSize(20, 20));
    }
    else
        printf("\nharr-cascades nao carregado\n");

    for(int i = 0 ; i < ( csFace ? csFace->total : 0 ) ; i++ ) {

        r = (CvRect*)cvGetSeqElem(csFace, i);

        cvRectangle( amFrame,
                     cvPoint( r->x, r->y ),
                     cvPoint( r->x + r->width, r->y + r->height ),
                     CV_RGB( 0, 255, 255 ),
                     2, 8, 0 );

       printf("\n face_x=%d face_y=%d wd=%d ht=%d",r->x,r->y,r->width,r->height);

    /** usar apenas no treinamento de VJ

      addlog(aimg_no, "FACE", csFace->total, i);

      **/

       detectEyes(amFrame,r,storage, aimg_no);
       cvResetImageROI(amFrame);
       detectMouth(amFrame, r, storage, aimg_no);
       cvResetImageROI(amFrame);

    }
    /** usar apenas em treinamento com RNA

        if (type == IS_FACE){
            itype_awnser = 1; }
        else{
            itype_awnser = 0;
        }

        storeAttainment(r->x, r->y, r->width, r->height, itype_awnser);
    **/

      cvReleaseHaarClassifierCascade( &lcFace );
      cvReleaseMemStorage( &storage );


}

IplImage* transformImgs(IplImage* amframe){

    double scale = 1.3;

    IplImage* gray = cvCreateImage(cvSize(amframe->width,
                                          amframe->height),
                                          8, 1 );

    IplImage* small_img = cvCreateImage(cvSize(cvRound(amframe->width/scale),
                                               cvRound(amframe->height/scale)),
                                               8, 1);

    cvCvtColor( amframe, gray, CV_BGR2GRAY );
    cvResize( gray, small_img, CV_INTER_LINEAR );

    cvEqualizeHist( small_img, small_img );
    cvSmooth(small_img, small_img, CV_GAUSSIAN);
    cvThreshold(small_img, small_img, 140, 255, CV_THRESH_BINARY);

    amframe = cvCloneImage(small_img);

    cvReleaseImage(&gray);
    cvReleaseImage(&small_img);

return amframe;

}
