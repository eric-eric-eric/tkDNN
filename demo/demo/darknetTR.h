#ifndef DEMO_H
#define DEMO_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include <malloc.h>
#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"
#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


/** PyImgReader
 *  a class that translate an array of bytes into an OpenCV Mat
*/
class PyImgReader
{
    public:
    /*the resulting image
    */
    cv::Mat img;

    /* contructor that receiver a pointer to data and makes
       it into an image.
       void* data is a pointer to the future image data
       int h is the desired image height
       int w is the desired image width
       int type is the pixel's type. One of OpenCV's
                constants, such as cv::CV_8UC3
    */
    PyImgReader(void* data, int h, int w, int type);

    /*Function that returns a pointer to the object's image
    */
    cv::Mat* getImg();
};

extern "C"
{
    typedef struct {
        int w;
        int h;
        int c;
        float *data;
    } image;

    typedef struct {
        float x, y, w, h;
    }BOX;

    typedef struct {
        int cl;
        BOX bbox;
        float prob;
        char name[20];

    }detection;

    tk::dnn::Yolo3Detection* load_network(char* net_cfg, int n_classes, int n_batch);

    /** Wrapper that returns a new PyImgReader
     *  Arguments are the same as in its constructor
     *
    */
    PyImgReader* PyImgR_new(void* data, int h, int w, int type)
    {
        return new PyImgReader(data, h, w, type);
    }

    /** Wrapper that receives a PyImgReader (obj)
     *  and returns a pointer to its image
    */
    cv::Mat* PyImgR_getImg(PyImgReader* obj)
    {
        return obj->getImg();
    }

    /** Wrapper that deletes a PyImgReader
    */
    void PyImgR_delete(PyImgReader* obj)
    {
        delete obj;
    }
}
#endif /* DETECTIONNN_H*/