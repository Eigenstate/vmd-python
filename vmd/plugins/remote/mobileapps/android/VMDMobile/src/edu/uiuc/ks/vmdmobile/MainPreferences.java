package edu.uiuc.ks.vmdmobile;
 
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.Preference;
import android.preference.PreferenceActivity;
import android.preference.Preference.OnPreferenceClickListener;
import android.widget.Toast;
 
public class MainPreferences extends PreferenceActivity {
   public static final String prefFile = "mainprefs";
   @Override
   protected void onCreate(Bundle savedInstanceState) {
      super.onCreate(savedInstanceState);
      getPreferenceManager().setSharedPreferencesName(prefFile);
      addPreferencesFromResource(R.layout.mainpreferences);

   }
}

