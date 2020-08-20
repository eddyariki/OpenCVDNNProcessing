import KinectPV2.*;
import java.nio.*;
import imageTranslater.*;
import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.*;
import org.opencv.ml.*;
import org.opencv.utils.*;
import org.opencv.video.*;
import org.opencv.videoio.*;
import java.text.DecimalFormat;

//Processing PImage to OpenCV mat translator from:

//https://sites.google.com/site/gutugutu30/other/matopencvtobufferedimagetopimageprocesingwoxianghubianhuankenengnaraiburariwozuottemita

//Kinect
KinectPV2 kinect;

//DNN config
String DIRECTORY = "ENTER DIRECTORY FOR DATA FOLDER HERE";
float THRESHOLD = 0.13;
Net net; 
int cols=512, rows=424;

//DNN results
ArrayList<int[]> faces = new ArrayList<int[]>();//Stores face position
ArrayList<String> confScore = new ArrayList<String>();//Stores confidence
ArrayList<PImage> faceGray = new ArrayList<PImage>();//Stores the grayscale Infrared images of each face

String text;
DecimalFormat df = new DecimalFormat(".00%");

int X_START = 277, X_END = 1726; //Constants used to map infrared image to color image;

//Image Output
int irX = 200, irY = 200;
PGraphics irOut;
int faceIndex1 = 0, faceIndex2=0;

void setup() {
  fullScreen(P2D,2);
  
  //Set up kinect
  kinect = new KinectPV2(this);
  kinect.enableInfraredImg(true);
  kinect.init();

  //Set up opencv native code
  System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

  //Model and prototex
  String model = DIRECTORY + "res10_300x300_ssd_iter_140000.caffemodel";
  String config = DIRECTORY + "deploy.prototxt.txt";

  //Create net 
  net = Dnn.readNetFromCaffe(config, model);
  //IR output
  irOut = createGraphics(512, 424, P2D);
  
}


void draw() {
  background(0,0,0);
    //----------------------------------------------------------------------
    //Face Detection
    //----------------------------------------------------------------------
    PImage infraredImage = kinect.getInfraredImage();
    //infraredImage.resize(400,0);
    PImage infraredImageEq = new PImage();
    if (infraredImage.width>1) {
      //Convert PImage of infrared to OpenCV mat type CV_U8C3
      Mat image = ImageTranslater.PImageToMat(infraredImage);
      Mat dst = new Mat();
  
      //Convert image to grayscale so we can equalize histogram
      Imgproc.cvtColor(image, dst, Imgproc.COLOR_RGB2GRAY);
      //dst.convertTo(dst,-1,1,-50+mouseX);
      //Equalizing histogram is needed with kinect's infrared feed
      Imgproc.equalizeHist(dst, dst);
      
      //Convert equalized grayscale back to RGB
      Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2RGB);
  
  
      //Save equalized grayscale to use later if needed
      infraredImageEq = ImageTranslater.MatToPImageRGB(this, dst);
  
      //Create blob
      Mat blob = Dnn.blobFromImage(dst, 1.0, new Size(300, 300), new Scalar(0., 0., 0.), false, false);
      net.setInput(blob);
      Mat result = new Mat();
  
      //Detect Faces
      try {
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        result = net.forward();
      }
      catch (Exception e) {
        println("DNN forward failed: " + e.getMessage());
      }
      //Prepare data to iterate over faces found
      result = result.reshape(1, (int)result.total() / 7);
  
      //Get the face position and confidence
      for (int i=0; i<result.rows(); i++) {
        double confidence = result.get(i, 2)[0];
        if (confidence > THRESHOLD) {
          int[] pos = new int[4];
          pos[0] = (int)(result.get(i, 3)[0] *cols);
          pos[1]    = (int)(result.get(i, 4)[0]*rows);
          pos[2]  = (int)(result.get(i, 5)[0]*cols);
          pos[3] = (int)(result.get(i, 6)[0]*rows);
          text = df.format(confidence);
          //Store the faces and confidence
          faces.add(pos);
          confScore.add(text);
        }
      }
    }
    faceGray.clear();
  
    extractFaces(faces, infraredImageEq.copy());
  
    if (frameCount%20==0) {
      faceIndex1 = int(random(0, faceGray.size()-1)); 
      faceIndex2 = int(random(0, faceGray.size()-1));
    }
  
    if (faceIndex1 > faceGray.size()-1) {
      faceIndex1 = int(random(0, faceGray.size()-1));
    }
    if (faceIndex2 > faceGray.size()-1) {
      faceIndex2 = int(random(0, faceGray.size()-1));
    }
  
  
  
    //Draw
    irOut.beginDraw();
      irOut.background(0);
      irOut.set(0,0,infraredImageEq);
      irOut.noFill();
      irOut.strokeWeight(3);
      if (faces.size()>0) {
        for (int i=0; i<faces.size(); i++) {
          irOut.noFill();
          irOut.stroke(215, 252, 3);
          irOut.rect(faces.get(i)[2], faces.get(i)[3], faces.get(i)[0]-faces.get(i)[2], faces.get(i)[1]-faces.get(i)[3]);
          irOut.textSize(16);
          irOut.fill(0, 255, 0);
          irOut.text(confScore.get(i), faces.get(i)[0], (faces.get(i)[1]-10+irY>10)? faces.get(i)[1]-10 : faces.get(i)[1]+10);
        }
      }
      faces.removeAll(faces);
      confScore.removeAll(confScore);
      irOut.endDraw();
  
    imageMode(CENTER);
    image(irOut,width/2,height/2,irOut.width*2.4,irOut.height*2.4);
    noFill();
    stroke(217, 255, 0);
}
