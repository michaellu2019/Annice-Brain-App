/*
CODE REFERENCES:
- https://www.allaboutcircuits.com/projects/control-an-arduino-using-your-phone/
- https://www.mirkosertic.de/blog/2013/07/realtime-face-detection-on-android-using-opencv/
- https://stackoverflow.com/questions/22769747/how-to-crop-mat-using-contours-in-opencv-for-java
- https://www.vladmarton.com/pocketsphinx-continuous-speech-recognition-android-tutorial/
*/

package com.example.littyling.annicebrain;

import android.Manifest;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.MediaPlayer;
import android.os.AsyncTask;
import android.os.CountDownTimer;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
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
import java.io.OutputStream;
import java.lang.ref.WeakReference;
import java.util.Set;
import java.util.UUID;

import edu.cmu.pocketsphinx.Assets;
import edu.cmu.pocketsphinx.Hypothesis;
import edu.cmu.pocketsphinx.RecognitionListener;
import edu.cmu.pocketsphinx.SpeechRecognizer;
import edu.cmu.pocketsphinx.SpeechRecognizerSetup;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2, RecognitionListener {
    // Logcat tag
    private static final String TAG = "Main";

    // vision processing
    private JavaCameraView javaCameraView;
    private ImageView imgView;
    private Mat mRgba;
    private Mat mGray;
    private CascadeClassifier faceCC;
    private CascadeClassifier upperBodyCC;
    private CascadeClassifier bodyCC;
    private CascadeClassifier catCC;
    private CascadeClassifier smileCC;
    private CascadeClassifier eyeCC;
    private CascadeClassifier bananaCC;
    private int absFaceSize;
    private int absUpperBodySize;
    private int absBodySize;
    private int absCatSize;
    private int absSmileSize;
    private int absEyeSize;
    private int absBananaSize;
    private int camWidth = 0;
    private int camHeight = 0;
    private String detectMode;

    // speech recognition with pocketsphinx
    private static String KWS_SEARCH = "wakeup";
    private static String MENU_SEARCH = "menu";
    private static String KEYPHRASE = "toby";
    private static SpeechRecognizer recognizer;

    // text to speech web view
    private WebView webView;
    private static String DEFAULT_VOICE = "UK English Male";
    private String lastMsg = "";

    // bluetooth
    private BluetoothAdapter btAdapter;
    private Set<BluetoothDevice> bondedDevices;
    private BluetoothDevice btDevice;
    private final String DEVICE_NAME = "HC-06";
    private final UUID PORT_UUID = UUID.fromString("0001101-0000-1000-8000-00805F9B34FB");
    private BluetoothSocket btSocket;
    private static OutputStream btOS;

    // mp3 media players
    private MediaPlayer beatsPlayer;
    private MediaPlayer nicePlayer;
    private MediaPlayer smilePlayer;
    private MediaPlayer introPlayer;
    private MediaPlayer frownPlayer;
    private MediaPlayer byePlayer;
    private MediaPlayer wayPlayer;
    private MediaPlayer ymcaPlayer;
    private MediaPlayer uptownPlayer;
    private MediaPlayer bgPlayer;
    private MediaPlayer aliveOnePlayer;
    private MediaPlayer aliveTwoPlayer;
    private MediaPlayer aliveThreePlayer;
    private MediaPlayer aliveFourPlayer;
    private MediaPlayer elevatorPlayer;
    private MediaPlayer brainPlayer;
    private MediaPlayer destructPlayer;
    private MediaPlayer bananaPlayer;
    private MediaPlayer sirPlayer;
    private MediaPlayer weePlayer;
    private MediaPlayer headPlayer;

    // program control flow
    private String command = "none";
    private boolean canListen = true;
    private int faceVal = 0;
    private final int FACE_THREHOLD = 20;
    private int smileVal = 0;
    private final int SMILE_THRESHOLD = 12;
    private int frownVal = 0;
    private final int FROWN_THRESHOLD = 500;

    // loads open cv elements
    BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS: {
                    // load classifiers
                    loadCascadeClassifiers("haarcascade_frontalface_default");
                    loadCascadeClassifiers("haarcascade_upperbody");
                    loadCascadeClassifiers("haarcascade_fullbody");
                    loadCascadeClassifiers("haarcascade_frontalcatface");
                    loadCascadeClassifiers("smile5");
                    loadCascadeClassifiers("haarcascade_eye");
                    loadCascadeClassifiers("banana_classifier");

                    // enable camera
                    javaCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    // debugs opencv load
    static {
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV successfully opened...");
        } else {
            Log.d(TAG, "OpenCV not successfully opened...");
        }
    }

    // load opencv cascade classifiers
    private void loadCascadeClassifiers(String fName) {
        try {
            // open xml file for classifier
            InputStream is = getResources().openRawResource(getResources().getIdentifier(fName, "raw", getPackageName()));
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, fName + ".xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            // read classifier and write out data
            byte[] buffer = new byte[4096];
            int bytesRead;

            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }

            is.close();
            os.close();

            // instantiate classifiers
            if (fName.equals("haarcascade_frontalface_default")) {
                faceCC = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            } else if (fName.equals("haarcascade_upperbody")) {
                upperBodyCC = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            } else if (fName.equals("haarcascade_fullbody")) {
                bodyCC = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            } else if (fName.equals("haarcascade_frontalcatface")) {
                catCC = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            } else if (fName.equals("smile5")) {
                smileCC = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            } else if (fName.equals("haarcascade_eye")) {
                eyeCC = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            } else if (fName.equals("banana_classifier")) {
                bananaCC = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            } else {
                faceCC = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            }

            Log.d(TAG, fName + " loaded successfully...");
        } catch (Exception e) {
            Log.e(TAG, "Error loading " + fName + "...", e);
        }
    }

    // sets up speech recognition
    private static class SetupTask extends AsyncTask<Void, Void, Exception> {
        WeakReference<MainActivity> activityReference;

        SetupTask(MainActivity activity) {
            this.activityReference = new WeakReference<>(activity);
        }

        @Override
        protected Exception doInBackground(Void... params) {
            try {
                Assets assets = new Assets(activityReference.get());
                File assetDir = assets.syncAssets();
                activityReference.get().setupRecognizer(assetDir);
            } catch (IOException e) {
                return e;
            }

            return null;
        }

        @Override
        protected void onPostExecute(Exception e) {
            if (e != null) {
                Log.e(TAG, "Speech recognition setup failed...\n" + e);
            } else {
                Log.d(TAG, "Speech recognition setup successfully...");
                activityReference.get().switchSearch(KWS_SEARCH);
            }
        }
    }

    private void setupRecognizer(File assetsDir) throws IOException {
        // setup speech recognizer object
        recognizer = SpeechRecognizerSetup.defaultSetup()
                .setAcousticModel(new File(assetsDir, "en-us-ptm"))
                .setDictionary(new File(assetsDir, "cmudict-en-us.dict"))
                .setBoolean("-remove_noise", true)
                .setKeywordThreshold(1e-20f)
                .setRawLogDir(assetsDir)
                .getRecognizer();
        recognizer.addListener(this);
        recognizer.addKeyphraseSearch(KWS_SEARCH, KEYPHRASE);

        File menuGrammar = new File(assetsDir, "mymenu.gram");
        recognizer.addGrammarSearch(MENU_SEARCH, menuGrammar);
    }

    // setup bluetooth
    private void startBluetooth() {
        btAdapter = BluetoothAdapter.getDefaultAdapter();
        bondedDevices = btAdapter.getBondedDevices();

        if(!btAdapter.isEnabled()) {
            Intent enableAdapter = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
            startActivityForResult(enableAdapter, 0);
        }

        if (bondedDevices.isEmpty()) {
            Toast.makeText(getApplicationContext(), "Pair the device...", Toast.LENGTH_SHORT).show();
        } else {
            for (BluetoothDevice bondedDevice : bondedDevices) {
                if (bondedDevice.getName().equals(DEVICE_NAME)) {
                    btDevice = bondedDevice;
                    break;
                }
            }
        }

        try {
            btSocket = btDevice.createRfcommSocketToServiceRecord(PORT_UUID);
            btSocket.connect();
            btOS = btSocket.getOutputStream();
            Log.d(TAG, "Bluetooth setup successfully...");
        } catch (Exception e) {
            Log.e(TAG, "Bluetooth setup failed...\n" + e);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Log.d(TAG, "Initiate " + TAG + "...");

        // setup camera and image view
        javaCameraView = (JavaCameraView) findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
        imgView = (ImageView) findViewById(R.id.image_view);

        // set detection focus
        detectMode = "face";

        // check for permission to record audio
        int permissionCheck = ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.RECORD_AUDIO);

        if (permissionCheck != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 1);
            return;
        }

        // setup speech recognition
        new SetupTask(this).execute();

        // setup bluetooth
        startBluetooth();
        btSendMsg("init", "0 0");

        // setup text to speech webview
        /*webView = (WebView) findViewById(R.id.web_view);
        webView.getSettings().setJavaScriptEnabled(true);
        webView.setWebChromeClient(new WebChromeClient());
        webView.loadUrl("file:///android_res/raw/index.html");*/

        // setup mp3 media players
        beatsPlayer = MediaPlayer.create(this, R.raw.beats);
        nicePlayer = MediaPlayer.create(this, R.raw.nice);
        smilePlayer = MediaPlayer.create(this, R.raw.smile);
        introPlayer = MediaPlayer.create(this, R.raw.smileintro);
        frownPlayer = MediaPlayer.create(this, R.raw.smilefrown);
        byePlayer = MediaPlayer.create(this, R.raw.smilehuman);
        wayPlayer = MediaPlayer.create(this, R.raw.way);
        ymcaPlayer = MediaPlayer.create(this, R.raw.ymca);
        uptownPlayer = MediaPlayer.create(this, R.raw.uptown);
        bgPlayer = MediaPlayer.create(this, R.raw.beegees);
        aliveOnePlayer = MediaPlayer.create(this, R.raw.aliveone);
        aliveTwoPlayer = MediaPlayer.create(this, R.raw.alivetwo);
        aliveThreePlayer = MediaPlayer.create(this, R.raw.alivethree);
        aliveFourPlayer = MediaPlayer.create(this, R.raw.alivefour);
        elevatorPlayer = MediaPlayer.create(this, R.raw.elevator);
        brainPlayer = MediaPlayer.create(this, R.raw.brain);
        destructPlayer = MediaPlayer.create(this, R.raw.destruct);
        bananaPlayer = MediaPlayer.create(this, R.raw.banana);
        sirPlayer = MediaPlayer.create(this, R.raw.sir);
        weePlayer = MediaPlayer.create(this, R.raw.wee);
        headPlayer = MediaPlayer.create(this, R.raw.myhead);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull  int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        // request permission to record audio
        if (requestCode == 1) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                new SetupTask(this).execute();
            } else {
                finish();
            }
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        // disable camera
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // disable camera
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }

        // shut down speech recognizer
        if (recognizer != null) {
            recognizer.cancel();
            recognizer.shutdown();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        // reload opencv
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV successfully opened...");
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "OpenCV not successfully opened...");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallBack);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        // init camera frame and settings
        camWidth = width;
        camHeight = height;
        mGray = new Mat(height, width, CvType.CV_8UC1);
        absFaceSize = (int) (height * 0.17);
        absUpperBodySize = (int) (height * 0.1);
        absBodySize = (int) (height * 0.1);
        absCatSize = (int) (height * 0.12);
        absSmileSize = (int) (height * 0.07);
        absEyeSize = (int) (height * 0.1);
        absBananaSize = (int) (height * 0.17);
    }

    @Override
    public void onCameraViewStopped() {
        // release frame
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        // set frame and convert to gray scale
        mRgba = inputFrame.rgba();
        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGBA2GRAY);
        MatOfRect objs = new MatOfRect();
        int objLimit = 1;
        boolean cropFin = false;
        boolean tracking = false;
        int offsetX = 0;
        int offsetY = 0;

        // detect respective objects
        if (detectMode.equals("face") && faceCC != null) {
            faceCC.detectMultiScale(mGray, objs, 1.1, 6, 0, new Size(absFaceSize, absFaceSize), new Size());
            tracking = true;
        } else if (detectMode.equals("upper body") && upperBodyCC != null) {
            upperBodyCC.detectMultiScale(mGray, objs, 1.05, 3, 0, new Size(absUpperBodySize, absUpperBodySize), new Size());
            cropFin = true;
        } else if (detectMode.equals("body") && bodyCC != null) {
            bodyCC.detectMultiScale(mGray, objs, 1.05, 2, 0, new Size(absBodySize, absBodySize), new Size());
            cropFin = true;
        } else if (detectMode.equals("cat") && catCC != null) {
            catCC.detectMultiScale(mGray, objs, 1.1, 6, 0, new Size(absCatSize, absCatSize), new Size());
            tracking = true;
        } else if (detectMode.equals("banana") && bananaCC != null) {
            bananaCC.detectMultiScale(mGray, objs, 1.1, 6, 0, new Size(absBananaSize, absBananaSize), new Size());
            tracking = true;
        } else if (detectMode.equals("smile") && smileCC != null) {
            // detect face to limit region of smile detection
            MatOfRect faces = new MatOfRect();
            faceCC.detectMultiScale(mGray, faces, 1.1, 6, 0, new Size(absFaceSize, absFaceSize), new Size());

            // detect smiles if faces present
            if (faces.toArray().length > 0) {
                Rect face = faces.toArray()[0];
                Mat faceROI = mGray.submat(new Rect(face.x, face.y, face.width, face.height));
                smileCC.detectMultiScale(faceROI, objs, 1.1, 20, 0, new Size(absSmileSize, absSmileSize), new Size());
                offsetX = face.x;
                offsetY = face.y;
                tracking = true;
            }
        } else if (detectMode.equals("eye") && eyeCC != null) {
            // detect face to limit region of eye detection
            MatOfRect faces = new MatOfRect();
            faceCC.detectMultiScale(mGray, faces, 1.1, 6, 0, new Size(absFaceSize, absFaceSize), new Size());

            // detect smiles if faces present
            if (faces.toArray().length > 0) {
                Rect face = faces.toArray()[0];
                Mat faceROI = mGray.submat(new Rect(face.x, face.y, face.width, (int) (face.height * 0.45)));
                eyeCC.detectMultiScale(faceROI, objs, 1.1, 6, 0, new Size(absEyeSize, absEyeSize), new Size());
                offsetX = face.x;
                offsetY = face.y;
                objLimit = 2;
            }
        }

        // render detections
        drawCenter(mRgba);
        drawBoxes(mRgba, objs.toArray(), objLimit, cropFin, tracking, offsetX, offsetY);

        return mRgba;
    }

    // draw center of screen
    public void drawCenter(Mat frame) {
        Imgproc.circle(frame, new Point(camWidth/2, camHeight/2), 5, new Scalar(255, 0, 0, 255), -1);
    }

    // draw bounding boxes around detected objects
    public void drawBoxes(Mat frame, Rect[] rects, int objLimit, boolean cropFin, boolean tracking, int offsetX, int offsetY) {
        for (int i = 0; i < rects.length; i++) {
            // limit objects rendered
            if (i >= objLimit)
                break;

            Rect rect = rects[i];

            // detection under commands
            if (command.equals("smile") && !canListen) {
                if (detectMode.equals("face")) {
                    if (smileVal == 0) {
                        // when face first encountered, and threshold passed greet person
                        if (faceVal > FACE_THREHOLD) {
                            if (!introPlayer.isPlaying()) {
                                introPlayer.seekTo(0);
                                introPlayer.start();
                                btSendMsg("speaking", "smile_intro");
                                Log.d(TAG, "Face threshold passed...");
                            } else {
                                // switch to smile detection after greeting
                                if (introPlayer.getCurrentPosition() > 8500) {
                                    detectMode = "smile";
                                    faceVal = 0;
                                }
                            }
                        } else {
                            faceVal++;
                            Log.d(TAG, "Face detected for " + faceVal + "...");
                        }
                    } else {
                        // if smile detection over, say bye
                        if (!byePlayer.isPlaying()) {
                            byePlayer.seekTo(0);
                            byePlayer.start();
                            btSendMsg("speaking", "smile_bye");
                            canListen = true;
                            Log.d(TAG, "Smile time is over? " + canListen + "...");
                        }
                    }
                } else if (detectMode.equals("smile")) {
                    // after greet, look for smile for some duration
                    if (smileVal > SMILE_THRESHOLD) {
                        // end smile detection after duration
                        Log.d(TAG, "Smile threshold passed...");
                        detectMode = "face";
                    } else {
                        smileVal++;
                        Log.d(TAG, "Smile detected for " + smileVal + "...");
                    }
                }
            }

            // bounding box
            if (detectMode.equals("upper body")) {
                Imgproc.rectangle(frame, rect.tl(), new Point(rect.br().x, rect.br().y + rect.height/2), new Scalar(0, 255, 0, 255), 3);
            } else if (detectMode.equals("smile") || detectMode.equals("eye")) {
                Rect newRect = new Rect((int) rect.tl().x + offsetX, (int) rect.tl().y + offsetY, rect.width, rect.height);
                rect = newRect;
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0, 255), 3);
            } else {
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0, 255), 3);
            }

            // center point of bounding box
            double centerX = rect.tl().x + (rect.br().x - rect.tl().x)/2;
            double centerY = rect.br().y - (rect.br().y - rect.tl().y)/2;

            Imgproc.circle(frame, new Point(centerX, centerY), 3, new Scalar(0, 255, 0, 255), -1);

            // calculate scaled values for how far bounding box center is from screen center
            int moveCeil = 20;
            double dX = centerX - camWidth/2;
            double dY = centerY - camHeight/2;
            int moveX = 0;
            int moveY = 0;
            int moveZ = 0;
            int zF = 10;

            if (Math.abs(dX) > moveCeil)
                moveX = -(int) (10 * dX/camWidth);

            if (Math.abs(dY) > moveCeil)
                moveY = -(int) (10 * dY/camWidth);

            if (rect.width < moveCeil * zF)
                moveZ = (int) (moveCeil * zF - rect.width)/5;

            if (tracking) {
                if (detectMode.equals("face") || detectMode.equals("smile")) {
                    btSendMsg("face_tracking", "" + moveX + " " + moveY);
                } else if (detectMode.equals("cat")) {
                    btSendMsg("cat_tracking", "" + moveX + " " + moveY + " " + moveZ);
                } else if (detectMode.equals("banana")) {
                    btSendMsg("banana_tracking", "" + moveX + " " + moveY + " " + moveZ);
                }
            }

            if (cropFin) {
                // crop image
                Mat imgCrop;

                if (detectMode.equals("upper body")) {
                    imgCrop = frame.submat(new Rect(rect.x, rect.y, rect.width, rect.height + rect.height/2));
                } else {
                    imgCrop = frame.submat(new Rect(rect.x, rect.y, rect.width, rect.height));
                }

                // display converted cropped image
                javaCameraView.setAlpha(0.2f);
                final Bitmap imgBM = Bitmap.createBitmap(imgCrop.cols(), imgCrop.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(imgCrop, imgBM);

                // idk what this really does but it's supposed to update image view with cropped image result
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        imgView.setImageBitmap(imgBM);
                    }
                });

                Log.d(TAG, "Image cropped...");
            }
        }

        // for cat detection, send empty values if no cat detected
        if (detectMode.equals("cat") && rects.length < 1)
            btSendMsg("cat_tracking", "" + 0 + " " + 0 + " " + 0);

        // for banana detection, send empty values if no banana detected
        if (detectMode.equals("banana") && rects.length < 1)
            btSendMsg("banana_tracking", "" + 0 + " " + 0 + " " + 0);

        // for smile detection, raise frown value if no smile detected and give reprimanding snarky comment
        if (command.equals("smile") && detectMode.equals("smile") && rects.length < 1) {
            if (frownVal > FROWN_THRESHOLD) {
                if (!frownPlayer.isPlaying() && !byePlayer.isPlaying()) {
                    frownPlayer.seekTo(0);
                    frownPlayer.start();
                    btSendMsg("speaking", "smile_frown");
                    Log.d(TAG, "Too much frown...");
                }
            } else {
                frownVal++;
                Log.d(TAG, "Frown detected for " + frownVal + "...");
            }

            if (frownPlayer.getCurrentPosition() > 3500 || byePlayer.isPlaying()) {
                frownVal = 0;
                frownPlayer.pause();
                frownPlayer.seekTo(0);
            }
        }
    }

    // handle speech
    public void handleCommand(String text) {
        if (text.equals(KEYPHRASE)) {
            // key phrase invokes listening for command
            switchSearch(MENU_SEARCH);
            btSendMsg("listening", "none");
            Log.d(TAG, "Listening...");
        } else if (command.equals("none")) {
            // initiates command
            if (text.equals("say hello")) {
                command = "greet";
                // speakMsg("greet", "Hi there; I'm Toby!", DEFAULT_VOICE);
                btSendMsg("speaking", "greet");

                command = "none";
                btSendMsg("speaking", "exit_command");
            } else if (text.equals("drop it")) {
                command = "uptown";
                btSendMsg("speaking", "uptown");

                if (!uptownPlayer.isPlaying()) {
                    uptownPlayer.seekTo(0);
                    uptownPlayer.start();
                }
            } else if (text.equals("smile time")) {
                command = "smile";
                detectMode = "face";
                faceVal = 0;
                smileVal = 0;
                frownVal = 0;
                canListen = false;
                btSendMsg("speaking", "smile");

                if (!smilePlayer.isPlaying()) {
                    smilePlayer.seekTo(0);
                    smilePlayer.start();
                }
            } else if (text.equals("party time")) {
                command = "party";
                btSendMsg("speaking","party");
                ymcaPlayer.seekTo(0);
                ymcaPlayer.start();
            } else if (text.equals("turn around")) {
                command = "turn";
                btSendMsg("speaking", "turn");

                if (!weePlayer.isPlaying()) {
                    weePlayer.seekTo(0);
                    weePlayer.start();
                }

                command = "none";
                btSendMsg("speaking", "exit_command");
            } else if (text.equals("stay still")) {
                command = "brain";
                btSendMsg("speaking", "brain");

                if (!brainPlayer.isPlaying()) {
                    brainPlayer.seekTo(0);
                    brainPlayer.start();
                }

                command = "none";
                btSendMsg("speaking", "exit_command");
            } else if (text.equals("cardboard")) {
                command = "head";
                btSendMsg("speaking", "head");

                if (!headPlayer.isPlaying()) {
                    headPlayer.seekTo(0);
                    headPlayer.start();
                }

                command = "none";
                btSendMsg("speaking", "exit_command");
            } else if (text.equals("hold on")) {
                command = "hold";
                btSendMsg("speaking", "hold");

                if (!elevatorPlayer.isPlaying()) {
                    elevatorPlayer.seekTo(0);
                    elevatorPlayer.start();
                }
            } else if (text.equals("are you ready")) {
                command = "ready";
                btSendMsg("speaking", "ready");

                if (!sirPlayer.isPlaying()) {
                    sirPlayer.seekTo(0);
                    sirPlayer.start();
                }

                command = "none";
                btSendMsg("speaking", "exit_command");
            } else if (text.equals("we are done")) {
                command = "done";
                btSendMsg("speaking", "done");

                if (!destructPlayer.isPlaying()) {
                    destructPlayer.seekTo(0);
                    destructPlayer.start();
                }

                command = "none";
                btSendMsg("speaking", "exit_command");
            } else if (text.equals("banana")) {
                command = "banana";
                btSendMsg("speaking", "banana");
                detectMode = "banana";

                if (!bananaPlayer.isPlaying()) {
                    bananaPlayer.seekTo(0);
                    bananaPlayer.start();
                }
            }

            /* else if (text.equals("thank you")) {
                command = "thanks";
                btSendMsg("speaking", "thanks");

                command = "none";
                btSendMsg("speaking", "exit_command");
            } else if (text.equals("lets have a conversation")) {
                command = "talk";
                speakMsg("talk", "Okay!", DEFAULT_VOICE);
                switchSearch(MENU_SEARCH);
            } else if (text.equals("ay pee ay pee ay pee")) {
                command = "ap";
                speakMsg("ap", "As always as usual, keep your eyes on your own paper; roving eyes will be penalized; answers should be concealed, not revealed; responses go on your test, not on your desk; and even on a depressing day like this, I can still spot. A cheetah; An agonizing pun. Ay. pee.", DEFAULT_VOICE);
            } else if (text.equals("dab on the haters")) {
                command = "dab";
                speakMsg("dab", "Dabbing on all the haters!", DEFAULT_VOICE);
            } else if (text.equals("drop it")) {
                if (!beatsPlayer.isPlaying()) {
                    command = "chill";
                    speakMsg("chill", "Dropping sick beats!", DEFAULT_VOICE);
                    beatsPlayer.seekTo(0);
                    beatsPlayer.start();
                } else {
                    speakMsg("chill", "We already chilling fam!", DEFAULT_VOICE);
                }
            } else if (text.equals("follow me")) {
                command = "follow";
                detectMode = "cat";
                btSendMsg("speaking", "follow");
                // speakMsg("follow", "Do yu no dah way dah way dah way?", DEFAULT_VOICE);
                if (!wayPlayer.isPlaying()) {
                    wayPlayer.seekTo(0);
                    wayPlayer.start();
                }
            } else if (text.equals("whats the meaning of life") || text.equals("what is the meaning of life")) {
                command = "meaning";
                speakMsg("meaning_of_life", "The meaning of life is to give life a meaning... Said no one ever.", DEFAULT_VOICE);
            }*/
        } else if (canListen && !command.equals("none")) {
            // handles command processes
            if (text.equals("exit command") || text.equals("that is enough")) {
                if (command.equals("smile time")) {
                    smilePlayer.pause();
                    introPlayer.pause();
                    frownPlayer.pause();
                    byePlayer.pause();
                }

                if (command.equals("follow"))
                    wayPlayer.pause();

                if (command.equals("chill"))
                    beatsPlayer.pause();

                if (command.equals("brain"))
                    brainPlayer.pause();

                if (command.equals("head"))
                    headPlayer.pause();

                if (command.equals("turn"))
                    weePlayer.pause();

                if (command.equals("ready"))
                    sirPlayer.pause();

                if (command.equals("party"))
                    ymcaPlayer.pause();

                if (command.equals("uptown"))
                    uptownPlayer.pause();

                if (command.equals("hold"))
                    elevatorPlayer.pause();

                if (command.equals("done"))
                    destructPlayer.pause();

                if (command.equals("banana"))
                    bananaPlayer.pause();

                if (command.equals("alive")) {
                    aliveOnePlayer.pause();
                    aliveTwoPlayer.pause();
                    aliveThreePlayer.pause();
                    aliveFourPlayer.pause();
                }

                detectMode = "face";

                command = "none";
                btSendMsg("speaking", "exit_command");
                // speakMsg("exit_command", "Exiting command!", DEFAULT_VOICE);
            }/*else if (command.equals("talk")) {
                if (text.equals("hello") || text.equals("hi")) {
                    speakMsg("hello","Hello there!", DEFAULT_VOICE);
                } else if (text.equals("goodbye") || text.equals("bye")) {
                    speakMsg("goodbye","Goodbye!", DEFAULT_VOICE);
                } else if (text.equals("whats your name") || text.equals("what is your name")) {
                    speakMsg("name","My name is Toby.", DEFAULT_VOICE);
                } else if (text.equals("how are you doing") || text.equals("how you doing") || text.equals("whats up") || text.equals("how are you")) {
                    speakMsg("okay", "I am okay.", DEFAULT_VOICE);
                }

                switchSearch(MENU_SEARCH);
            }*/
        } else {
            Log.d(TAG, "Logging: " + text);
        }
    }

    @Override
    public void onPartialResult(Hypothesis hypothesis) {
        if (hypothesis != null)
            handleCommand(hypothesis.getHypstr());
    }

    @Override
    public void onResult(Hypothesis hypothesis) {
//        if (hypothesis != null)
//            handleCommand(hypothesis.getHypstr());
    }

    @Override
    public void onBeginningOfSpeech() {
        // custom action on beginning of speech
    }

    @Override
    public void onEndOfSpeech() {
        // reset speech recognizer to listen
        if (!recognizer.getSearchName().equals(KWS_SEARCH))
            switchSearch(KWS_SEARCH);
    }

    private void switchSearch(String searchName) {
        // switch between recognition of phrases or menu items within timeout of 10 seconds
        recognizer.stop();

        if (searchName.equals(KWS_SEARCH)) {
            recognizer.startListening(searchName);
        } else {
            recognizer.startListening(searchName, 10000);
        }
    }

    @Override
    public void onError(Exception e) {
        // print errors
        Log.e(TAG, "Error encountered...\n" + e);
    }

    @Override
    public void onTimeout() {
        // switch back to listening if no speech received
        switchSearch(KWS_SEARCH);
    }

    // send text to speech message
    private void speakMsg(String msgId, String msg, String voice) {
        if (!msg.equals(lastMsg)) {
            webView.evaluateJavascript("javascript:speak('" + msg + "', '" + voice + "')", null);
            btSendMsg("speaking", msgId);
            Log.d(TAG, "Speak: " + msg + " in " + voice + "...");

            lastMsg = msg;
        }
    }

    // write bluetooth message
    public void btSendMsg(String id, String msg) {
        if (btOS == null)
            return;

        String finalMsg = "<" + id + " " + msg + ">";
        try {
            btOS.write(finalMsg.getBytes());
            Log.d(TAG, finalMsg);
        } catch (Exception e) {
            Log.e(TAG, "Error writing to bluetooth output stream...\n" + e);
        }
    }

    // add delay in program
    public void delay(int timeout) {
        try {
            Thread.sleep(timeout);
        } catch (Exception e) {
            Log.e(TAG, "Error in delay:\n" + e);
        }
    }
}
