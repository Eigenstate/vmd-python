package edu.uiuc.ks.vmdmobile;
 
import android.app.Activity;
import android.app.Dialog;
import android.app.ListActivity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.Preference;
import android.preference.PreferenceActivity;
import android.preference.Preference.OnPreferenceClickListener;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.Toast;
 
public class ServerChooser extends ListActivity {
   final String tag = "ServerChooser";
   String [] names;
   ArrayAdapter ad;
   public static final String SERVER_NAMES = "serverList";

   private VMDMobile m_mob;
   private ServerChooser thisObj = this;

   final int MAX_LIST_ELEMENTS = 10;

   @Override
   protected void onCreate(Bundle savedInstanceState) {
      super.onCreate(savedInstanceState);
      setContentView(R.layout.serverchooser);

      // let's get the main app
      // not actually getting any data from the main class right now
//      Bundle b = getIntent().getExtras();
//      m_mob = (VMDMobile) b.get("app");

      SharedPreferences list = getSharedPreferences(SERVER_NAMES, MODE_PRIVATE);

      int numServers = list.getInt("NumServers", 0);

      names = new String[numServers + 1];
      names[0] = "(Add New VMD Server...)";
      for (int i=0; i<numServers; i++)
      {
         names[i+1] = list.getString("server" + i, "ERROR");
      }

      registerForContextMenu(getListView());

      // Create an ArrayAdapter, that will actually make the Strings above
      // appear in the ListView
      // (simple_list_item_1 is a generic value that will get created because of
      // this call)?
      ad = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1,
                                          names);
      this.setListAdapter(ad);
   }

   // ------------------------------------------------------------------------

   @Override
   protected void onListItemClick(ListView l, View v, int position, long id) {
      super.onListItemClick(l, v, position, id);

      // did they click item 0, the one to 'find a new VMD server' ?

      if (position == 0)
      {
         showNewServerDialog();
      }
      else
      {
         SharedPreferences list = 
                             getSharedPreferences(SERVER_NAMES, MODE_PRIVATE);

         moveServerListItem(list, position-1, 0);
         // they chose the most recently used one anyway, so
         // let's just use it.
//         m_mob.selectNewServer(
//               list.getString("server0", "ERROR"),
//               list.getInt("port0", VMDMobile.DEFAULT_VMD_PORT),
//               list.getInt("listen0", VMDMobile.DEFAULT_LISTEN_PORT));
         Intent i = new Intent(thisObj, VMDMobile.class);
//         Bundle b = new Bundle();
//         b.putString("serverName",  list.getString("server0", "ERROR"));
//         b.putInt("port", list.getInt("port0", VMDMobile.DEFAULT_VMD_PORT));
//         b.putInt("listen", 
//                       list.getInt("listen0", VMDMobile.DEFAULT_LISTEN_PORT));
//         i.putExtras(b);
         Log.d(tag, "onListItemClick.sending list item");
         i.putExtra("serverName",  list.getString("server0", "ERROR"));
         i.putExtra("port", list.getInt("port0", VMDMobile.DEFAULT_VMD_PORT));
         i.putExtra("listen", 
                       list.getInt("listen0", VMDMobile.DEFAULT_LISTEN_PORT));
         setResult(Activity.RESULT_OK,i);
         finish();
//         startActivity(i);
      }
//      // Figure out which item was clicked
//      Object o = this.getListAdapter().getItem(position);
//      String keyword = o.toString();
//      Toast.makeText(this, "You selected " + position + ": " + keyword,
//                                              Toast.LENGTH_SHORT)
//            .show();
//      names[position] = names[position] + "x";
//      ad.notifyDataSetChanged();
   }  // end of onListItemClick()

   // ------------------------------------------------------------------------
   private void moveServerListItem(SharedPreferences list, 
                                            final int oldLoc, final int newLoc)
   {
      if (oldLoc == newLoc) return; // nothing to do

      // save old location data
      String sTmp = list.getString("server" + oldLoc, "ERROR");
      int pTmp = list.getInt("port" + oldLoc, VMDMobile.DEFAULT_VMD_PORT);
      int lTmp = list.getInt("listen" + oldLoc, VMDMobile.DEFAULT_LISTEN_PORT);

      SharedPreferences.Editor ed = list.edit();

      // move the items down, one at a time
      for (int i=oldLoc; i>newLoc; i--)
      {
         ed.putString("server" + i, list.getString("server" + (i-1), "ERROR"));
         ed.putInt("port" + i, 
                  list.getInt("port" + (i-1), VMDMobile.DEFAULT_VMD_PORT));
         ed.putInt("listen" + i, 
                  list.getInt("listen" + (i-1), VMDMobile.DEFAULT_LISTEN_PORT));
      }
      // and copy the saved temporary data back into the desired location
      ed.putString("server" + newLoc, sTmp);
      ed.putInt("port" + newLoc, pTmp);
      ed.putInt("listen" + newLoc, lTmp);

      ed.commit();  // save it!
   }

   // ------------------------------------------------------------------------
   private void saveToServerList(final String strServer, final int serverPort,
                                      final int listenPort)
   {
      SharedPreferences list = 
                             getSharedPreferences(SERVER_NAMES, MODE_PRIVATE);
      int numServers = list.getInt("NumServers", 0);

      int desiredLocation = Math.min(numServers, MAX_LIST_ELEMENTS-1);

      SharedPreferences.Editor ed = list.edit();
      ed.putString("server" + desiredLocation, strServer);
      ed.putInt("port" + desiredLocation, serverPort);
      ed.putInt("listen" + desiredLocation, listenPort);
      if (numServers < MAX_LIST_ELEMENTS) {
         ed.putInt("NumServers", numServers+1);
      }
      ed.commit();  // save it!

      // move that one into spot zero
      moveServerListItem(list, desiredLocation, 0);

      Log.d(tag, "saveToServerList: server:" + strServer + ",port:" +
                  serverPort + ",listen:" + listenPort + ",numServers:" +
                  numServers + ", desiredLoc:" + desiredLocation);
   }

   // ----------------------------------------------------------------------
   private void showNewServerDialog()
   {
      Log.d(tag, "showNewServerDialog().start");
      // custom dialog
      final Dialog dialog = new Dialog(this);
      dialog.setContentView(R.layout.addnewserverinformation);
 
      // if button is clicked, close the custom dialog
      Button dialogButton = (Button) dialog.findViewById(R.id.dialogButtonCancel);
      dialogButton.setOnClickListener(new OnClickListener() {
         @Override
         public void onClick(View v) {
            dialog.dismiss();
         }
      });
      dialogButton = (Button) dialog.findViewById(R.id.dialogButtonOK);
      dialogButton.setOnClickListener(new OnClickListener() {
         @Override
         public void onClick(View v) {
            String strServer = ((EditText)
               dialog.findViewById(R.id.ipEntry)).getText().toString();
            if (strServer.equals("")) {
               Toast.makeText(thisObj, "You need to enter a server name",
                                              Toast.LENGTH_LONG).show();
            } else {
               String s = ((EditText)
                     dialog.findViewById(R.id.portEntry)).getText().toString();
               int serverPort = VMDMobile.DEFAULT_VMD_PORT;
               try
               {
                  serverPort = Integer.parseInt(s);
               } catch (NumberFormatException e) {}
               s = ((EditText)
                dialog.findViewById(R.id.listenportEntry)).getText().toString();
               int listenPort = VMDMobile.DEFAULT_LISTEN_PORT;
               try
               {
                  listenPort = Integer.parseInt(s);
               } catch (NumberFormatException e) {}

// Let's save this to the active list...........
               saveToServerList(strServer, serverPort, listenPort);

//               m_mob.selectNewServer(strServer, serverPort, listenPort);
               Log.d(tag, "showNewServerDialog. sending new server item");
               Intent i = new Intent(thisObj, VMDMobile.class);
//               Bundle b = new Bundle();
//               b.putString("serverName", strServer);
//               b.putInt("port", serverPort);
//               b.putInt("listen", listenPort);
               i.putExtra("serverName", strServer);
               i.putExtra("port", serverPort);
               i.putExtra("listen", listenPort);
//               i.putExtras(b);
               setResult(Activity.RESULT_OK,i);
//               startActivity(i);
               dialog.dismiss();
               finish();
            }
         }
      });
 
      Log.d(tag, "showNewServerDialog().before show()");
      dialog.show();
   }


/*

// save the new values

SharedPreferences list = getSharedPreferences(SERVER_NAMES, MODE_PRIVATE);
SharedPreferences.Editor ed = settings.edit();
ed.put blah ("name" "value");

ed.commit();
*/

}
