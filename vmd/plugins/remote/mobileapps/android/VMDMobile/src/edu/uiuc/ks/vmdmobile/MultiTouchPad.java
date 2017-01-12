package edu.uiuc.ks.vmdmobile;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.AttributeSet;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.PorterDuff;
import android.view.InputDevice;
import android.view.MotionEvent;
import android.view.View;
import org.metalev.multitouch.controller.MultiTouchController;
import org.metalev.multitouch.controller.MultiTouchController.MultiTouchObjectCanvas;
import org.metalev.multitouch.controller.MultiTouchController.PointInfo;
import org.metalev.multitouch.controller.MultiTouchController.PositionAndScale;

// basic class taken from a code.google.com bug comment and tweaked
// heavily

public class MultiTouchPad extends View implements MultiTouchObjectCanvas {

   private MultiTouchController multiTouchController = 
                                      new MultiTouchController(this);

   private PointInfo currTouchPoints = new PointInfo();

   private Paint mTouchCircleActive = new Paint();
   private Paint mTouchCircleInActive = new Paint();
   private Paint mTouchCircleStartPoint = new Paint();
  private Bitmap img;
  private float x = 100;
  private float y = 100;;
  private float scale = 1;
  private float angle = 0;
  private VMDMobile app;
  SharedPreferences mPrefs = null;
  private boolean m_InStroke = false;

  private float startX, startY, startPressure;

// ------------------------------------------------------------
private Bitmap adjustOpacity(Bitmap bitmap, int opacity)
{
    Bitmap mutableBitmap = bitmap.isMutable()
                           ? bitmap
                           : bitmap.copy(Bitmap.Config.ARGB_8888, true);
    Canvas canvas = new Canvas(mutableBitmap);
    int colour = (opacity & 0xFF) << 24;
    canvas.drawColor(colour, PorterDuff.Mode.DST_IN);
    return mutableBitmap;
}

   // ------------------------------------------------------------

   private void init()
   {

      img = adjustOpacity(BitmapFactory.decodeResource(getResources(), 
                                       R.drawable.ic_launcher_fullsize), 60);

      mTouchCircleActive.setColor(Color.YELLOW);
      mTouchCircleActive.setStrokeWidth(5);
      mTouchCircleActive.setStyle(Style.STROKE);
      mTouchCircleActive.setAntiAlias(true);

      mTouchCircleStartPoint.setColor(Color.GRAY);
      mTouchCircleStartPoint.setStrokeWidth(5);
      mTouchCircleStartPoint.setStyle(Style.STROKE);
      mTouchCircleStartPoint.setAntiAlias(true);

      mTouchCircleInActive.setColor(Color.RED);
      mTouchCircleInActive.setStrokeWidth(3);
      mTouchCircleInActive.setStyle(Style.STROKE);
      mTouchCircleInActive.setAntiAlias(true);



      setBackgroundColor(Color.BLACK);
   }


   // ------------------------------------------------------------
   public MultiTouchPad(Context context, AttributeSet a) {
    super(context, a);
    init();
   }


   // ------------------------------------------------------------
   public MultiTouchPad(Context context) {
    super(context);
    init();
   }

   // ------------------------------------------------------------
   public void registerApp(VMDMobile app)
   {
      this.app = app;
      final String prefFile = MainPreferences.prefFile;
      mPrefs = app.getSharedPreferences(prefFile, Context.MODE_PRIVATE);
   }

   // ------------------------------------------------------------
   private void setTouchPoints(PointInfo ctp)
   {
      currTouchPoints.set(ctp);
      app.registerTouch(currTouchPoints);
   }

   // ------------------------------------------------------------

   @Override
   protected void onDraw(Canvas canvas) {
      super.onDraw(canvas);
//      System.out.println("starting in onDraw" +
//            "this.getWidth(): " + this.getWidth() + ", " +
//            "this.getHeight(): " + this.getHeight() + ", " +
//            "canvas.getWidth(): " + canvas.getWidth() + ", " +
//            "canvas.getHeight(): " + canvas.getHeight() + ", " +
//            "img.getWidth(): " + img.getWidth() + ", " +
//            "img.getHeight(): " + img.getHeight() + ", " +
//            "img.getScaledWidth(): " + img.getScaledWidth(canvas) + ", " +
//            "img.getScaledHeight(): " + img.getScaledHeight(canvas) );

    if (mPrefs.getBoolean("showBackgroundImage",true)) {
       canvas.save();
       canvas.scale(
               0.9f * this.getWidth() / img.getScaledWidth(canvas),
               0.9f * this.getHeight() /  img.getScaledHeight(canvas)
   );
       canvas.drawBitmap(img, x, y, null);
       canvas.restore();
    }
    drawMultitouchDebugMarks(canvas);
   } // end of onDraw

   // ------------------------------------------------------------

   private void drawMultitouchDebugMarks(Canvas canvas) {
      Paint mDraw;
      if (currTouchPoints.isDown()) {

         float[] xs = currTouchPoints.getXs();
         float[] ys = currTouchPoints.getYs();
         float[] pressures = currTouchPoints.getPressures();
         int numPoints = currTouchPoints.getNumTouchPoints();

         if (app.sendingTouch())
         {
            mDraw = mTouchCircleActive;

            // are we in the middle of a stroke?  If so, let's mark
            // the beginning point
            if (numPoints == 1) {
               if (m_InStroke) {
                  canvas.drawCircle(startX, startY, 50 + startPressure * 80,
                                           mTouchCircleStartPoint);
               } else {
                  m_InStroke = true;
                  // we should be starting a stroke
                  startX = xs[0];
                  startY = ys[0];
                  startPressure = pressures[0];
               }
            } else {
               m_InStroke = false;
            }
         } else {
            mDraw = mTouchCircleInActive;
         }
         for (int i = 0; i < numPoints; i++)
         {
            canvas.drawCircle(xs[i], ys[i], 50 + pressures[i] * 80,
                                           mDraw);
            // foobar.  Send an event here, or do a callback
         }
         if (numPoints == 2)
         {
            canvas.drawLine(xs[0], ys[0], xs[1], ys[1],
                                           mDraw);
         }
      } else {
         // no fingers are down right now
         m_InStroke = false;

      }
   }

   // ------------------------------------------------------------

   /** Pass touch events to the MT controller */
   @Override
   public boolean onTouchEvent(MotionEvent event) {
      InputDevice id = event.getDevice();
      app.logConsole(
         "Action:" + event.getAction() +
         ",AIndex:" + event.getActionIndex() +
         ",AMasked:" + event.getActionMasked() +
         ",BState:" + event.getButtonState() +
         ",DeviceId:" + event.getDeviceId() +
         ",Flags:" + event.getFlags() +
         ",Meta:" + event.getMetaState() +
//         ",PCount:" + event.getPounterCount() +
//         ",PId:" + event.getPounterId() +
         ",Src:" + event.getSource() 
         );
//      System.out.println("starting in onTouchEvent");
      return multiTouchController.onTouchEvent(event);
   }

   /** Get the image that is under the single-touch point, or return null
(canceling the drag op) if none */
   public Object getDraggableObjectAtPoint(PointInfo pt) {
//      System.out.println("starting in getDraggableObjectAtPoint");
    return this;
   }

   /**
    * Select an object for dragging. Called whenever an object is found to be
under the point (non-null is returned by getDraggableObjectAtPoint())
    * and a drag operation is starting. Called with null when drag op ends.
    */
   public void selectObject(Object obj, PointInfo touchPoint) {
//      System.out.println("starting in selectObject. tp: " + touchPoint.info());
//      System.out.print("X");
      setTouchPoints(touchPoint);
      invalidate();
   }

   /** Get the current position and scale of the selected image. Called whenever
a drag starts or is reset. */
   public void getPositionAndScale(Object obj, PositionAndScale
objPosAndScaleOut) {
//      System.out.println("starting in getPositionAndScale");
//      objPosAndScaleOut.set(x, y, true, scale, false, scale, scale,  true, angle);
   }

   /** Set the position and scale of the dragged/stretched image. */
   public boolean setPositionAndScale(Object obj, PositionAndScale
newPosAndScale, PointInfo touchPoint) {
//      System.out.println("starting in SetPositionAndScale. tp:" + touchPoint.info());
//      System.out.print("x");
      setTouchPoints(touchPoint);
//    x = newPosAndScale.getXOff();
//    y = newPosAndScale.getYOff();
//    scale = newPosAndScale.getScale();
//    angle = newPosAndScale.getAngle();
    invalidate();
      return true;
   }
}

