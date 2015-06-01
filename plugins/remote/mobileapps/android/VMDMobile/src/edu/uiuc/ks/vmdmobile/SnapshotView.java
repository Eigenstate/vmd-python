package edu.uiuc.ks.vmdmobile;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.os.AsyncTask;
import android.os.Environment;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

import java.io.File;
import java.io.InputStream;
import java.io.BufferedInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;


import org.metalev.multitouch.controller.MultiTouchController;
import org.metalev.multitouch.controller.MultiTouchController.MultiTouchObjectCanvas;
import org.metalev.multitouch.controller.MultiTouchController.PointInfo;
import org.metalev.multitouch.controller.MultiTouchController.PositionAndScale;


public class SnapshotView extends View implements MultiTouchObjectCanvas {

   private MultiTouchController multiTouchController = new MultiTouchController(this);

   private PointInfo currTouchPoints = new PointInfo();

//   private Paint mLinePaintTouchPointCircle = new Paint();
  private Bitmap img;
  private float x = 0;
  private float y = 0;
  private boolean initialScaleSet = false;
  private float scale = 1;
  private float totalScale = 1;
  private float maxScaleAllowed = 2;
  private float angle = 0;
  private boolean allowRotate = true;
  private Paint mPaint;
  private boolean m_bWaiting = false;

// ------------------------------------------------------------------
   public Bitmap getBitmap() {
     return img;
   }

// ------------------------------------------------------------------
   private void initialization() 
   {
      Log.d("SnapshotView", "initialization. begin");
     img = BitmapFactory.decodeResource(getResources(), R.drawable.noimage);
//    img = BitmapFactory.decodeResource(getResources(), R.drawable.a);
//    setImage("http://www.ks.uiuc.edu/~kvandivo/9_30party-1.JPG");
//    setImage("http://www.ks.uiuc.edu/~kvandivo/a.bmp");
//    setImage("http://athine.ks.uiuc.edu:5141/7384");

     x = 0;
     y = 0;
     initialScaleSet = false;

     angle = 0;
     mPaint = new Paint();

     mPaint.setDither(true);
//     mPaint.setAntiAlias(false);
     mPaint.setAntiAlias(true);
     mPaint.setFilterBitmap(true);

     setBackgroundColor(Color.BLACK);

   }
// ------------------------------------------------------------------
   public SnapshotView(Context context) {
     super(context);
     initialization();
   }

// ------------------------------------------------------------------
   public SnapshotView(Context context, AttributeSet as) {
     super(context, as);
     initialization();
   }


// ------------------------------------------------------------------
   public void allowRotate(final boolean b) {
      allowRotate = b;
   }
// ------------------------------------------------------------------

 

   @Override
   protected void onDraw(Canvas canvas) {
      super.onDraw(canvas);
      canvas.save();

      if (!initialScaleSet) {
         int imgW = img.getWidth();
         int imgH = img.getHeight();
         int viewW = this.getWidth();
         int viewH = this.getHeight();
         float scaleX = (viewW + 0.0f) / imgW;
         float scaleY = (viewH + 0.0f) / imgH;
         scale = Math.min(scaleX, scaleY);
         Log.d("SnapshotView", "onDraw." +
               "imgW:" + imgW + "imgH:" + imgH +
               "viewW:" + viewW + "viewH:" + viewH +
               "scaleX:" + scaleX + "scaleY:" + scaleY +
               "scale:" + scale);
         initialScaleSet = true;
      }

      float scaleAmount = scale; 

     Log.d("SnapshotView", "onDraw. scale:" + scale);

/*
      if (totalScale * scaleAmount > maxScaleAllowed) {
         scaleAmount = maxScaleAllowed / totalScale;
      }

      totalScale *= scaleAmount;
*/
//      scaleAmount = Math.min(maxScaleAllowed, scaleAmount);

//      Log.d("SnapshotView", "onDraw. scale" + scale + ",x: " + x + ",y: " + y);
      // scale the same amount in x and y, pivoting about point (x,y)
      canvas.scale(scaleAmount, scaleAmount, x, y);

      if (allowRotate) canvas.rotate(angle * 180.0f / (float) Math.PI, x, y);


      // img will be placed with its left edge at x, top edge at y
      canvas.drawBitmap(img, x, y, mPaint);
      canvas.restore();
   }

// ------------------------------------------------------------------

   /** Pass touch events to the MT controller */
   @Override
   public boolean onTouchEvent(MotionEvent event) {
      return multiTouchController.onTouchEvent(event);
   }

   /** Get the image that is under the single-touch point, or return null
(canceling the drag op) if none */
   public Object getDraggableObjectAtPoint(PointInfo pt) {
    return this;
   }

   /**
    * Select an object for dragging. Called whenever an object is found to be
under the point (non-null is returned by getDraggableObjectAtPoint())
    * and a drag operation is starting. Called with null when drag op ends.
    */
   public void selectObject(Object obj, PointInfo touchPoint) {
      currTouchPoints.set(touchPoint);
      invalidate();
   }

   /** Get the current position and scale of the selected image. Called whenever
a drag starts or is reset. */
   public void getPositionAndScale(Object obj, 
                                   PositionAndScale objPosAndScaleOut) 
   {
      objPosAndScaleOut.set(x, y, true, scale, false, scale, scale,  true,
angle);
   }

   /** Set the position and scale of the dragged/stretched image. */
   public boolean setPositionAndScale(Object obj, 
                                      PositionAndScale newPosAndScale, 
                                      PointInfo touchPoint) 
   {
      currTouchPoints.set(touchPoint);
      x = newPosAndScale.getXOff();
      y = newPosAndScale.getYOff();
      scale = newPosAndScale.getScale();
//      Log.d("SnapshotView", "setPositionAndScale. scale" + scale);
      angle = newPosAndScale.getAngle();
      invalidate();
      return true;
   }

   // ----------------------------------------------------------------
   class GetImage extends AsyncTask<String, Void, Object>
   {
      protected Object doInBackground(String... parms)
      {
      URL myFileUrl =null;          
      try {
         myFileUrl= new URL(parms[0]);
      } catch (MalformedURLException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
         Log.d("SnapshotView:GetImage", "URL create error is " + e);
      }

      // get the filename...
      String [] strList = myFileUrl.getPath().split("/");
      String filename = strList[ strList.length - 1];

      String strOut = getContext().getCacheDir() + "/" + filename;

      try {
         HttpURLConnection conn=
                    (HttpURLConnection)myFileUrl.openConnection();
         conn.setDoInput(true);
         conn.connect();
         InputStream is = conn.getInputStream();

         BufferedInputStream inStream = new BufferedInputStream(is, 1024 * 5);
         FileOutputStream outStream = new FileOutputStream(strOut);
         byte[] buff = new byte[5 * 1024];

         //Read bytes (and write them) until there is nothing more to read(-1)
         int len;
         while ((len = inStream.read(buff)) != -1)
         {
            outStream.write(buff,0,len);
         }

         //clean up
         outStream.flush();
         outStream.close();
         inStream.close();

         // have to do this in two steps (download and then decodeFile)
         // because android can't read a BMP from a stream and figure
         // out what to do with it. 
         img = BitmapFactory.decodeFile(strOut);
         (new File(strOut)).delete();
      } catch (IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
         Log.d("SnapshotView:GetImage", "Error is " + e);
      }
//         Log.d("SnapshotView:GetImage", "end of doInBackground");
         m_bWaiting = false;
         return null;
      } // end of doInBackground

   } // end of GetImage class

   // ----------------------------------------------------------------
// ------------------------------------------------------------------
   public void setImage(String url) {
      m_bWaiting = true;
      new GetImage().execute(url);
      while (m_bWaiting) {
         try {
            Thread.sleep(100);
         } catch (Exception e) {} 
      }
   } // end of setImage
}

