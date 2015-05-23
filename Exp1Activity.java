package com.example.adamjiang.faceacupoint;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;

import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.Toast;
import android.widget.ZoomControls;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import com.polites.android.GestureImageView;
import com.google.android.glass.touchpad.Gesture;
import com.google.android.glass.touchpad.GestureDetector;
public class Exp1Activity extends Activity implements CvCameraViewListener2{

    GestureImageView image;
    private static final String TAG  = "OCVSample::Activity";
    private static final Scalar RED = new Scalar(255, 0, 0, 255);
    private static final Scalar GREEN = new Scalar(0, 255, 0, 255);
    private static final Scalar BLUE = new Scalar(0, 0, 255, 255);
    private final static Scalar BLACK = new Scalar(0, 0, 0, 0);
    private Mat temp;
    private Mat mRgba;
    private Mat mGray;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    int currentZoomLevel = 0, maxZoomLevel = 0;
    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;
    private Camera.Parameters params;
    private Camera camera;
    private CameraBridgeViewBase mOpenCvCameraView;
    //JavaCameraView mycamera;
    public volatile int zoomlevel=11;
    GestureDetector mGestureDetector;
    FrameLayout frameLayout;


    private static final Point[] mapFace = { //total: 53
            new Point(0.5,0.3),new Point(0.5,0.7),new Point(0.5,0.6),new Point(0.5,0.78),new Point(0.5,0.86),new Point(0.37,0.9),new Point(0.63,0.9),new Point(0.65,0.87),
            new Point(0.35,0.87),new Point(0.3,0.85),new Point(0.7,0.85),new Point(0.22,0.74),new Point(0.78,0.74),new Point(0.17,0.55),new Point(0.83,0.55),new Point(0.145,0.45),
            new Point(0.855,0.45),new Point(0.4,0.5),new Point(0.6,0.5),new Point(0.65,0.82),new Point(0.35,0.82),new Point(0.64,0.65),new Point(0.36,0.65),new Point(0.63,0.5),
            new Point(0.37,0.5),new Point(0.62,0.45),new Point(0.38,0.45),new Point(0.74,0.6),new Point(0.26,0.6),new Point(0.6,0.62),new Point(0.4,0.62),new Point(0.87,0.56),
            new Point(0.13,0.56),new Point(0.88,0.52),new Point(0.12,0.52),new Point(0.89,0.48),new Point(0.11,0.48),new Point(0.73,0.44),new Point(0.27,0.44),new Point(0.6,0.38),
            new Point(0.4,0.38),new Point(0.79,0.38),new Point(0.21,0.38),new Point(0.87,0.31),new Point(0.13,0.31),new Point(0.59,0.29),new Point(0.41,0.29),new Point(0.69,0.25),
            new Point(0.31,0.25),new Point(0.79,0.27),new Point(0.21,0.27),new Point(0.695,0.14),new Point(0.305,0.14)
    }; // This is the regex that (mostly) compressed the repeated code: /.*tl\(\)\.x\+(0\.[0-9]+)\*\(faces.*tl\(\)\.y\+(0\.[0-9]+)\*\(faces.*/
    private static final String[] mapFacePinyin = { //total: 53
            "YinTang","ShuiGou","SuLiao","DuiDuan","ChengJiang","DaYing","DaYing","JiaChengJiang",
            "JiaChengJiang","JiaChe","JiaChe","QianZheng","QianZheng","XiaGuan","XiaGuan","ShangGuan",
            "ShangGuan","BiTong","BiTong","DiCang","DiCang","JuLiao","JuLiao","SiBai",
            "SiBai","ChengQi","ChengQi","Guan","Guan","YingXiang","YingXiang","TingHui",
            "TingHui","TingGong","TingGong","ErMen","ErMen","QiuHou","QiuHou","QingMing",
            "QingMing","TongZi","TongZi","TaiYang","TaiYang","ZanZhu","ZanZhu","YuYao",
            "YuYao","SiKongZhu","SiKongZhu","YangBai","YangBai"
    }; //Names of the above points, in pinyin (formerly the variable names of the points).
    private int frameNumber = 0;

    private TextAwesome text1;

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

    public Exp1Activity()
    {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.exp1);

        mGestureDetector = createGestureDetector(this);
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
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }



    private GestureDetector createGestureDetector(Context context) {
        GestureDetector gestureDetector = new GestureDetector(context);

        //Create a base listener for generic gestures
        gestureDetector.setBaseListener( new GestureDetector.BaseListener() {
            @Override
            public boolean onGesture(Gesture gesture) {
                if (gesture == Gesture.TAP) {
                    zoomlevel = 1;
                    return true;
                } else if (gesture == Gesture.TWO_TAP) {

                    zoomlevel = 11;
                    return true;
                } else if (gesture == Gesture.SWIPE_RIGHT) {
                    if (zoomlevel>2)
                    {
                        zoomlevel = zoomlevel-2;
                    }
                    return true;
                } else if (gesture == Gesture.SWIPE_LEFT) {
                    if (zoomlevel<10)
                    {
                        zoomlevel = zoomlevel+2;
                    }
                    return true;
                } else if (gesture == Gesture.SWIPE_DOWN){
                    Exp1Activity.this.finish();
                    return true;
                }
                return false;
            }
        });

        gestureDetector.setFingerListener(new GestureDetector.FingerListener() {
            @Override
            public void onFingerCountChanged(int previousCount, int currentCount) {
            }
        });

        gestureDetector.setScrollListener(new GestureDetector.ScrollListener() {
            @Override
            public boolean onScroll(float displacement, float delta, float velocity) {
                return true;
            }
        });

        return gestureDetector;
    }

    @Override
    public boolean onGenericMotionEvent(MotionEvent event) {
        if (mGestureDetector != null) {
            return mGestureDetector.onMotionEvent(event);
        }
        return false;
    }
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Size sizeRgba = mRgba.size();
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;

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
        for (int i = 0; i < facesArray.length; i++){

            Log.i(TAG, "+++++++++++++++facesArray: number: "+facesArray.length);
            try {
//                Rect h = new Rect(
//                        new Point(rows / 2 - zoomlevel * rows / 100, cols / 2 - zoomlevel * cols / 100),
//                        new Point(rows / 2 + zoomlevel * rows / 100, cols / 2 + zoomlevel * cols / 100));
                Mat mZoomWindow = mRgba.submat(rows / 2 - zoomlevel * rows / 100, rows / 2 + zoomlevel * rows / 100, cols / 2 - zoomlevel * cols / 100, cols / 2 + zoomlevel * cols / 100);
                String[] rystr = new String[mapFace.length]; //buffer for file writing
                for (int j = mapFace.length; j-- > 0; ) {
                    Point pt = new Point(facesArray[i].tl().x + mapFace[j].x * facesArray[i].width, facesArray[i].tl().y + mapFace[j].y * facesArray[i].height);
                    //Core.rectangle(mRgba, pt, pt, FACE_RECT_COLOR, 3);
                    //now with crosshairs
                    int length = 2;
                    Core.line(mRgba, new Point(pt.x - length, pt.y), new Point(pt.x + length, pt.y), BLUE, 1);
                    Core.line(mRgba, new Point(pt.x, pt.y - length), new Point(pt.x, pt.y + length), BLUE, 1);
                    if (zoomlevel<5){
                        pt.x = pt.x - 20;
                        Core.putText(mRgba, mapFacePinyin[j], pt, Core.FONT_HERSHEY_PLAIN, 0.5, BLACK, 1);

                    }

                    rystr[j] = frameNumber + "," + j + "," + pt.x + "," + pt.y + "\n";
                }
                frameNumber++;

//region Points
//                Point YinTang = new Point(facesArray[i].tl().x+0.5*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.3*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, YinTang, YinTang, FACE_RECT_COLOR, 3);
//
//                Point ShuiGou = new Point(facesArray[i].tl().x+0.5*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.7*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ShuiGou, ShuiGou, FACE_RECT_COLOR, 3);
//
//                Point SuLiao = new Point(facesArray[i].tl().x+0.5*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.6*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, SuLiao, SuLiao, FACE_RECT_COLOR, 3);
//
//                Point DuiDuan = new Point(facesArray[i].tl().x+0.5*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.78*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, DuiDuan, DuiDuan, FACE_RECT_COLOR, 3);
//
//                Point ChengJiang = new Point(facesArray[i].tl().x+0.5*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.86*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ChengJiang, ChengJiang, FACE_RECT_COLOR, 3);
//
//                Point DaYing_left = new Point(facesArray[i].tl().x+0.37*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.9*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, DaYing_left, DaYing_left, FACE_RECT_COLOR, 3);
//
//                Point DaYing_right = new Point(facesArray[i].tl().x+0.63*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.9*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, DaYing_right, DaYing_right, FACE_RECT_COLOR, 3);
//
//                Point JiaChengJiang_right = new Point(facesArray[i].tl().x+0.65*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.87*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, JiaChengJiang_right, JiaChengJiang_right, FACE_RECT_COLOR, 3);
//
//                Point JiaChengJiang_left = new Point(facesArray[i].tl().x+0.35*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.87*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, JiaChengJiang_left, JiaChengJiang_left, FACE_RECT_COLOR, 3);
//
//                Point JiaChe_left = new Point(facesArray[i].tl().x+0.3*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.85*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, JiaChe_left, JiaChe_left, FACE_RECT_COLOR, 3);
//
//                Point JiaChe_right = new Point(facesArray[i].tl().x+0.7*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.85*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, JiaChe_right, JiaChe_right, FACE_RECT_COLOR, 3);
//
//                Point QianZheng_left = new Point(facesArray[i].tl().x+0.22*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.74*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, QianZheng_left, QianZheng_left, FACE_RECT_COLOR, 3);
//
//                Point QianZheng_right = new Point(facesArray[i].tl().x+0.78*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.74*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, QianZheng_right, QianZheng_right, FACE_RECT_COLOR, 3);
//
//                Point XiaGuan_left = new Point(facesArray[i].tl().x+0.17*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.55*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, XiaGuan_left, XiaGuan_left, FACE_RECT_COLOR, 3);
//
//                Point XiaGuan_right = new Point(facesArray[i].tl().x+0.83*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.55*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, XiaGuan_right, XiaGuan_right, FACE_RECT_COLOR, 3);
//
//                Point ShangGuan_left = new Point(facesArray[i].tl().x+0.145*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.45*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ShangGuan_left, ShangGuan_left, FACE_RECT_COLOR, 3);
//
//                Point ShangGuan_right = new Point(facesArray[i].tl().x+0.855*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.45*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ShangGuan_right, ShangGuan_right, FACE_RECT_COLOR, 3);
//
//                Point BiTong_left = new Point(facesArray[i].tl().x+0.4*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.5*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, BiTong_left, BiTong_left, FACE_RECT_COLOR, 3);
//
//                Point BiTong_right = new Point(facesArray[i].tl().x+0.6*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.5*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, BiTong_right, BiTong_right, FACE_RECT_COLOR, 3);
//
//                Point DiCang_right = new Point(facesArray[i].tl().x+0.65*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.82*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, DiCang_right, DiCang_right, FACE_RECT_COLOR, 3);
//
//                Point DiCang_left = new Point(facesArray[i].tl().x+0.35*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.82*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, DiCang_left, DiCang_left, FACE_RECT_COLOR, 3);
//
//                Point JuLiao_right = new Point(facesArray[i].tl().x+0.64*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.65*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, JuLiao_right, JuLiao_right, FACE_RECT_COLOR, 3);
//
//                Point JuLiao_left = new Point(facesArray[i].tl().x+0.36*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.65*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, JuLiao_left, JuLiao_left, FACE_RECT_COLOR, 3);
//
//                Point SiBai_right = new Point(facesArray[i].tl().x+0.63*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.5*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, SiBai_right, SiBai_right, FACE_RECT_COLOR, 3);
//
//                Point SiBai_left = new Point(facesArray[i].tl().x+0.37*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.5*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, SiBai_left, SiBai_left, FACE_RECT_COLOR, 3);
//
//                Point ChengQi_right = new Point(facesArray[i].tl().x+0.62*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.45*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ChengQi_right, ChengQi_right, FACE_RECT_COLOR, 3);
//
//                Point ChengQi_left = new Point(facesArray[i].tl().x+0.38*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.45*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ChengQi_left, ChengQi_left, FACE_RECT_COLOR, 3);
//
//                Point Guan_right = new Point(facesArray[i].tl().x+0.74*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.6*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, Guan_right, Guan_right, FACE_RECT_COLOR, 3);
//
//                Point Guan_left = new Point(facesArray[i].tl().x+0.26*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.6*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, Guan_left, Guan_left, FACE_RECT_COLOR, 3);
//
//                Point YingXiang_right = new Point(facesArray[i].tl().x+0.6*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.62*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, YingXiang_right, YingXiang_right, FACE_RECT_COLOR, 3);
//
//                Point YingXiang_left = new Point(facesArray[i].tl().x+0.4*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.62*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, YingXiang_left, YingXiang_left, FACE_RECT_COLOR, 3);
//
//                Point TingHui_right = new Point(facesArray[i].tl().x+0.87*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.56*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, TingHui_right, TingHui_right, FACE_RECT_COLOR, 3);
//
//                Point TingHui_left = new Point(facesArray[i].tl().x+0.13*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.56*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, TingHui_left, TingHui_left, FACE_RECT_COLOR, 3);
//
//                Point TingGong_right = new Point(facesArray[i].tl().x+0.88*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.52*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, TingGong_right, TingGong_right, FACE_RECT_COLOR, 3);
//
//                Point TingGong_left = new Point(facesArray[i].tl().x+0.12*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.52*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, TingGong_left, TingGong_left, FACE_RECT_COLOR, 3);
//
//                Point ErMen_right = new Point(facesArray[i].tl().x+0.89*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.48*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ErMen_right, ErMen_right, FACE_RECT_COLOR, 3);
//
//                Point ErMen_left = new Point(facesArray[i].tl().x+0.11*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.48*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ErMen_left, ErMen_left, FACE_RECT_COLOR, 3);
//
//                Point QiuHou_right = new Point(facesArray[i].tl().x+0.73*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.44*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, QiuHou_right, QiuHou_right, FACE_RECT_COLOR, 3);
//
//                Point QiuHou_left = new Point(facesArray[i].tl().x+0.27*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.44*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, QiuHou_left, QiuHou_left, FACE_RECT_COLOR, 3);
//
//                Point QingMing_right = new Point(facesArray[i].tl().x+0.6*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.38*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, QingMing_right, QingMing_right, FACE_RECT_COLOR, 3);
//
//                Point QingMing_left = new Point(facesArray[i].tl().x+0.4*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.38*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, QingMing_left, QingMing_left, FACE_RECT_COLOR, 3);
//
//                Point TongZi_right = new Point(facesArray[i].tl().x+0.79*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.38*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, TongZi_right, TongZi_right, FACE_RECT_COLOR, 3);
//
//                Point TongZi_left = new Point(facesArray[i].tl().x+0.21*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.38*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, TongZi_left, TongZi_left, FACE_RECT_COLOR, 3);
//
//                Point TaiYang_right = new Point(facesArray[i].tl().x+0.87*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.31*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, TaiYang_right, TaiYang_right, FACE_RECT_COLOR, 3);
//
//                Point TaiYang_left = new Point(facesArray[i].tl().x+0.13*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.31*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, TaiYang_left, TaiYang_left, FACE_RECT_COLOR, 3);
//
//                Point ZanZhu_right = new Point(facesArray[i].tl().x+0.59*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.29*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ZanZhu_right, ZanZhu_right, FACE_RECT_COLOR, 3);
//
//                Point ZanZhu_left = new Point(facesArray[i].tl().x+0.41*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.29*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, ZanZhu_left, ZanZhu_left, FACE_RECT_COLOR, 3);
//
//                Point YuYao_right = new Point(facesArray[i].tl().x+0.69*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.25*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, YuYao_right, YuYao_right, FACE_RECT_COLOR, 3);
//
//                Point YuYao_left = new Point(facesArray[i].tl().x+0.31*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.25*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, YuYao_left, YuYao_left, FACE_RECT_COLOR, 3);
//
//                Point SiKongZhu_right = new Point(facesArray[i].tl().x+0.79*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.27*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, SiKongZhu_right, SiKongZhu_right, FACE_RECT_COLOR, 3);
//
//                Point SiKongZhu_left = new Point(facesArray[i].tl().x+0.21*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.27*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, SiKongZhu_left, SiKongZhu_left, FACE_RECT_COLOR, 3);
//
//                Point YangBai_right = new Point(facesArray[i].tl().x+0.695*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.14*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, YangBai_right, YangBai_right, FACE_RECT_COLOR, 3);
//
//                Point YangBai_left = new Point(facesArray[i].tl().x+0.305*(facesArray[i].br().x-facesArray[i].tl().x), facesArray[i].tl().y+0.14*(facesArray[i].br().y-facesArray[i].tl().y));
//
//                Core.rectangle(mRgba, YangBai_left, YangBai_left, FACE_RECT_COLOR, 3);
//endregion
                Log.i(TAG, rows + "&" + cols);
                Mat zoomCorner = mRgba.submat(0, rows / 2 - rows*5 / 100, 0, cols / 2 - cols*5 / 100);
                //Mat zoomCorner = mRgba.submat(0, rows*3 / 4, 0, cols*4 / 5);


                Imgproc.resize(mZoomWindow, zoomCorner, zoomCorner.size());
                //Size wsize = mZoomWindow.size();
                //Core.rectangle(mZoomWindow, new Point(1, 1), new Point(wsize.width - 2, wsize.height - 2), new Scalar(255, 0, 0, 255), 2);
                //mRgba = zoomCorner;
                //zoomCorner.copyTo(mRgba);
                zoomCorner.release();
                mZoomWindow.release();
                save(rystr); //use the buffer
            }
                catch (Exception e){Log.d("Exception",e.getMessage());}
        }

        return mRgba;
    }
    private void save(String[] rystr) {

        for (String str : rystr) {
            Log.i(TAG, str);
        }
    }
//    private void initFileSaving(){
//        Log.i(TAG, "Starting to save the file.");
//        DateFormat dateFormat = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US);
//        dataFileName = dateFormat.format(new Date()) + "_acupt.txt";
//        try{
//            FileOutputStream outputStream = openFileOutput(dataFileName, Context.MODE_APPEND);
//            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
//            outputStreamWriter.write("Frame number,Point number,x,y\n");
//            outputStreamWriter.close();
//        } catch (IOException e){
//            e.printStackTrace();
//        }
}
