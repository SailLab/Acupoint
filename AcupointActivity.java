package com.example.adamjiang.faceacupoint;


import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.Features2d;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;

import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import java.io.IOException;
public class AcupointActivity extends Activity implements CvCameraViewListener2{
   // Filter myFace = null;
    private static final String TAG  = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR  = new Scalar(0, 0, 255, 128);
    private int n=0;
    private Mat mRgba;
    private Mat mGray;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    //private final Mat mReferenceImage;
    private MatOfKeyPoint mReferenceKeypoints;
    private Mat mReferenceDescriptors;
    // CvType defines the color depth, number of channels, and
    // channel layout in the image. Here, each point is represented
    // by two 32-bit floats.
    private Mat mReferenceCorners;

    private MatOfKeyPoint mSceneKeypoints;
    private Mat mSceneDescriptors;
    private Mat mCandidateSceneCorners;
    private Mat mSceneCorners;
    private MatOfPoint mIntSceneCorners;

    private Mat mGraySrc;
    private MatOfDMatch mMatches;

    private FeatureDetector mFeatureDetector;
    private DescriptorExtractor mDescriptorExtractor;
    private DescriptorMatcher mDescriptorMatcher;

    private final Scalar mLineColor = new Scalar(0, 255, 0);
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);
                        //myFace = new FaceDetectionFilter( AcupointActivity.this, R.drawable.starry_night);
                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                    //   mOpenCvCameraView.enableFpsMeter();

                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public AcupointActivity()
    {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.detection_activity);


        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surfaceView);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        mReferenceKeypoints = new MatOfKeyPoint();
        mReferenceDescriptors = new Mat();
        mReferenceCorners =  new Mat(4, 1, CvType.CV_32FC2);

        mSceneKeypoints =  new MatOfKeyPoint();
        mSceneDescriptors = new Mat();
        mCandidateSceneCorners =  new Mat(4, 1, CvType.CV_32FC2);
        mSceneCorners = new Mat(4, 1, CvType.CV_32FC2);
        mIntSceneCorners = new MatOfPoint();

        mGraySrc = new Mat();
        mMatches = new MatOfDMatch();

        mFeatureDetector =  FeatureDetector.create(FeatureDetector.ORB);
        mDescriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        mDescriptorMatcher = DescriptorMatcher.create( DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }


    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();

        if (mJavaDetector != null)
            mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());

        Rect[] facesArray = faces.toArray();


        if (facesArray.length>0&&n==0)
        {
            n=1;
            final Mat referenceImageGray = new Mat();
            Imgproc.cvtColor(mRgba, referenceImageGray, Imgproc.COLOR_BGR2GRAY);
            mReferenceCorners.put(0, 0,  new double[] {50, 50});
            mReferenceCorners.put(1, 0,  new double[] {referenceImageGray.cols()-50, 50});
            mReferenceCorners.put(2, 0,  new double[] {referenceImageGray.cols()-50,  referenceImageGray.rows()-50});
            mReferenceCorners.put(3, 0,  new double[] {50, referenceImageGray.rows()-50});
//            mReferenceCorners.put(0, 0,  new double[] {facesArray[0].tl().x+0.5*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.3*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(1, 0,  new double[] {facesArray[0].tl().x+0.5*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.7*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(2, 0,  new double[] {facesArray[0].tl().x+0.5*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.6*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(3, 0,  new double[] {facesArray[0].tl().x+0.5*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.78*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(4, 0,  new double[] {facesArray[0].tl().x+0.5*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.86*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(5, 0,  new double[] {facesArray[0].tl().x+0.37*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.9*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(6, 0,  new double[] {facesArray[0].tl().x+0.63*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.9*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(7, 0,  new double[] {facesArray[0].tl().x+0.65*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.87*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(8, 0,  new double[] {facesArray[0].tl().x+0.35*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.87*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(9, 0,  new double[] {facesArray[0].tl().x+0.3*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.85*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(10, 0,  new double[] {facesArray[0].tl().x+0.7*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.85*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(11, 0,  new double[] {facesArray[0].tl().x+0.22*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.74*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(12, 0,  new double[] {facesArray[0].tl().x+0.78*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.74*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(13, 0,  new double[] {facesArray[0].tl().x+0.17*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.55*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(14, 0,  new double[] {facesArray[0].tl().x+0.83*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.55*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(15, 0,  new double[] {facesArray[0].tl().x+0.145*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.45*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(16, 0,  new double[] {facesArray[0].tl().x+0.855*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.45*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(17, 0,  new double[] {facesArray[0].tl().x+0.4*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.5*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(18, 0,  new double[] {facesArray[0].tl().x+0.6*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.5*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(19, 0,  new double[] {facesArray[0].tl().x+0.65*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.82*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(20, 0,  new double[] {facesArray[0].tl().x+0.35*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.82*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(21, 0,  new double[] {facesArray[0].tl().x+0.64*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.65*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(22, 0,  new double[] {facesArray[0].tl().x+0.36*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.65*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(23, 0,  new double[] {facesArray[0].tl().x+0.63*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.5*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(24, 0,  new double[] {facesArray[0].tl().x+0.37*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.5*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(25, 0,  new double[] {facesArray[0].tl().x+0.62*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.45*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(26, 0,  new double[] {facesArray[0].tl().x+0.38*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.45*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(27, 0,  new double[] {facesArray[0].tl().x+0.74*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.6*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(28, 0,  new double[] {facesArray[0].tl().x+0.26*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.6*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(29, 0,  new double[] {facesArray[0].tl().x+0.6*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.62*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(30, 0,  new double[] {facesArray[0].tl().x+0.4*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.62*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(31, 0,  new double[] {facesArray[0].tl().x+0.87*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.56*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(32, 0,  new double[] {facesArray[0].tl().x+0.13*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.56*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(33, 0,  new double[] {facesArray[0].tl().x+0.88*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.52*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(34, 0,  new double[] {facesArray[0].tl().x+0.12*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.52*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(35, 0,  new double[] {facesArray[0].tl().x+0.89*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.48*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(36, 0,  new double[] {facesArray[0].tl().x+0.11*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.48*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(37, 0,  new double[] {facesArray[0].tl().x+0.73*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.44*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(38, 0,  new double[] {facesArray[0].tl().x+0.27*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.44*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(39, 0,  new double[] {facesArray[0].tl().x+0.6*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.38*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(40, 0,  new double[] {facesArray[0].tl().x+0.4*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.38*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(41, 0,  new double[] {facesArray[0].tl().x+0.79*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.38*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(42, 0,  new double[] {facesArray[0].tl().x+0.21*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.38*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(43, 0,  new double[] {facesArray[0].tl().x+0.87*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.31*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(44, 0,  new double[] {facesArray[0].tl().x+0.13*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.31*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(45, 0,  new double[] {facesArray[0].tl().x+0.59*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.29*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(46, 0,  new double[] {facesArray[0].tl().x+0.41*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.29*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(47, 0,  new double[] {facesArray[0].tl().x+0.69*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.25*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(48, 0,  new double[] {facesArray[0].tl().x+0.31*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.25*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(49, 0,  new double[] {facesArray[0].tl().x+0.79*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.27*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(50, 0,  new double[] {facesArray[0].tl().x+0.21*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.27*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(51, 0,  new double[] {facesArray[0].tl().x+0.695*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.14*(facesArray[0].br().y-facesArray[0].tl().y)});
//            mReferenceCorners.put(52, 0,  new double[] {facesArray[0].tl().x+0.305*(facesArray[0].br().x-facesArray[0].tl().x), facesArray[0].tl().y+0.14*(facesArray[0].br().y-facesArray[0].tl().y)});

            mFeatureDetector.detect(referenceImageGray, mReferenceKeypoints);
            mDescriptorExtractor.compute(referenceImageGray,  mReferenceKeypoints, mReferenceDescriptors);
        }
        else if (facesArray.length>0&&n==1)
        {
            Imgproc.cvtColor(mRgba, mGraySrc, Imgproc.COLOR_RGBA2GRAY);
            mFeatureDetector.detect(mGraySrc, mSceneKeypoints);
            //Features2d.drawKeypoints(mGraySrc, mSceneKeypoints, mRgba);
            mDescriptorExtractor.compute(mGraySrc, mSceneKeypoints,  mSceneDescriptors);
            mDescriptorMatcher.match(mSceneDescriptors, mReferenceDescriptors, mMatches);
            findSceneCorners();
            draw(mRgba, mRgba);
        }
        else if (facesArray.length==0)
        {
            n=0;
        }

        for (int i = 0; i < facesArray.length; i++){
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);

        }
        return mRgba;
    }
    private void findSceneCorners() {

        final List<DMatch> matchesList = mMatches.toList();
        if (matchesList.size() < 4) {
            // There are too few matches to find the homography.
            return;
        }

        final List<KeyPoint> referenceKeypointsList = mReferenceKeypoints.toList();
        final List<KeyPoint> sceneKeypointsList =  mSceneKeypoints.toList();

        // Calculate the max and min distances between keypoints.
        double maxDist = 0.0;
        double minDist = Double.MAX_VALUE;
        for (final DMatch match : matchesList) {
            final double dist = match.distance;
            if (dist < minDist) {
                minDist = dist;
            }
            if (dist > maxDist) {
                maxDist = dist;
            }
        }

        if (minDist > 50.0) {
            // The target is completely lost.
            // Discard any previously found corners.
            mSceneCorners.create(0, 0, mSceneCorners.type());
            return;
        } else if (minDist > 25.0) {
            // The target is lost but maybe it is still close.
            // Keep any previously found corners.
            return;
        }

        // Identify "good" keypoints based on match distance.
        final ArrayList<Point> goodReferencePointsList = new ArrayList<Point>();
        final ArrayList<Point> goodScenePointsList = new ArrayList<Point>();
        final double maxGoodMatchDist = 1.75 * minDist;
        for (final DMatch match : matchesList) {
            if (match.distance < maxGoodMatchDist) {
                goodReferencePointsList.add(referenceKeypointsList.get(match.trainIdx).pt);
                goodScenePointsList.add( sceneKeypointsList.get(match.queryIdx).pt);
            }
        }

        if (goodReferencePointsList.size() < 4 ||  goodScenePointsList.size() < 4) {
            // There are too few good points to find the homography.
            return;
        }

        final MatOfPoint2f goodReferencePoints = new MatOfPoint2f();
        goodReferencePoints.fromList(goodReferencePointsList);

        final MatOfPoint2f goodScenePoints = new MatOfPoint2f();
        goodScenePoints.fromList(goodScenePointsList);

        final Mat homography = Calib3d.findHomography( goodReferencePoints, goodScenePoints);
        Core.perspectiveTransform(mReferenceCorners,  mCandidateSceneCorners, homography);

        mCandidateSceneCorners.convertTo(mIntSceneCorners, CvType.CV_32S);
        if (Imgproc.isContourConvex(mIntSceneCorners)) {
            mCandidateSceneCorners.copyTo(mSceneCorners);
        }
    }

    protected void draw(final Mat src, final Mat dst) {

        if (dst != src) {
            src.copyTo(dst);
        }
        Core.line(dst, new Point(mSceneCorners.get(0, 0)), new Point(mSceneCorners.get(1, 0)), mLineColor, 4);
        Core.line(dst, new Point(mSceneCorners.get(1, 0)), new Point(mSceneCorners.get(2, 0)), mLineColor, 4);
        Core.line(dst, new Point(mSceneCorners.get(2, 0)), new Point(mSceneCorners.get(3, 0)), mLineColor, 4);
        Core.line(dst, new Point(mSceneCorners.get(3, 0)), new Point(mSceneCorners.get(0, 0)), mLineColor, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(0, 0)), new Point(mSceneCorners.get(0, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(1, 0)), new Point(mSceneCorners.get(1, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(2, 0)), new Point(mSceneCorners.get(2, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(3, 0)), new Point(mSceneCorners.get(3, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(4, 0)), new Point(mSceneCorners.get(4, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(5, 0)), new Point(mSceneCorners.get(5, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(6, 0)), new Point(mSceneCorners.get(6, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(7, 0)), new Point(mSceneCorners.get(7, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(8, 0)), new Point(mSceneCorners.get(8, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(9, 0)), new Point(mSceneCorners.get(9, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(10, 0)), new Point(mSceneCorners.get(10, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(11, 0)), new Point(mSceneCorners.get(11, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(12, 0)), new Point(mSceneCorners.get(12, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(13, 0)), new Point(mSceneCorners.get(13, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(14, 0)), new Point(mSceneCorners.get(14, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(15, 0)), new Point(mSceneCorners.get(15, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(16, 0)), new Point(mSceneCorners.get(16, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(17, 0)), new Point(mSceneCorners.get(17, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(18, 0)), new Point(mSceneCorners.get(18, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(19, 0)), new Point(mSceneCorners.get(19, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(20, 0)), new Point(mSceneCorners.get(20, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(21, 0)), new Point(mSceneCorners.get(21, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(22, 0)), new Point(mSceneCorners.get(22, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(23, 0)), new Point(mSceneCorners.get(23, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(24, 0)), new Point(mSceneCorners.get(24, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(25, 0)), new Point(mSceneCorners.get(25, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(26, 0)), new Point(mSceneCorners.get(26, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(27, 0)), new Point(mSceneCorners.get(27, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(28, 0)), new Point(mSceneCorners.get(28, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(29, 0)), new Point(mSceneCorners.get(29, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(30, 0)), new Point(mSceneCorners.get(30, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(31, 0)), new Point(mSceneCorners.get(31, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(32, 0)), new Point(mSceneCorners.get(32, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(33, 0)), new Point(mSceneCorners.get(33, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(34, 0)), new Point(mSceneCorners.get(34, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(35, 0)), new Point(mSceneCorners.get(35, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(36, 0)), new Point(mSceneCorners.get(36, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(37, 0)), new Point(mSceneCorners.get(37, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(38, 0)), new Point(mSceneCorners.get(38, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(39, 0)), new Point(mSceneCorners.get(39, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(40, 0)), new Point(mSceneCorners.get(40, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(41, 0)), new Point(mSceneCorners.get(41, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(42, 0)), new Point(mSceneCorners.get(42, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(43, 0)), new Point(mSceneCorners.get(43, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(44, 0)), new Point(mSceneCorners.get(44, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(45, 0)), new Point(mSceneCorners.get(45, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(46, 0)), new Point(mSceneCorners.get(46, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(47, 0)), new Point(mSceneCorners.get(47, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(48, 0)), new Point(mSceneCorners.get(48, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(49, 0)), new Point(mSceneCorners.get(49, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(50, 0)), new Point(mSceneCorners.get(50, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(51, 0)), new Point(mSceneCorners.get(51, 0)), FACE_RECT_COLOR, 4);
//        Core.rectangle(dst, new Point(mSceneCorners.get(52, 0)), new Point(mSceneCorners.get(52, 0)), FACE_RECT_COLOR, 4);
    }
}
