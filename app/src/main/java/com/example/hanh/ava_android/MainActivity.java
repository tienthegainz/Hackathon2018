package com.example.hanh.ava_android;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.GoogleApiAvailability;
import com.google.android.gms.common.GooglePlayServicesUtil;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.MapFragment;
import com.google.android.gms.maps.MapView;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;

import java.util.logging.Logger;

import butterknife.BindView;
import butterknife.ButterKnife;

public class MainActivity extends AppCompatActivity implements OnMapReadyCallback {

    private GoogleMap map;
    private LocationManager locationManager;
    private static final String TAG = "MainActivity";
    private View view;
    private static final int LOCATION_UPDATE_MIN_TIME = 5000;
    private static final int LOCATION_UPDATE_MIN_DISTANCE = 10;
    private  MapFragment mapFragment;

    @BindView(R.id.btnActionFloat)
    FloatingActionButton btnFLoat;
    EditText edtDestination;
    @BindView(R.id.imgClearDesti)
    ImageView imgClear;

    private LocationListener locationListener = new LocationListener() {
        @SuppressLint("DefaultLocale")
        @Override
        public void onLocationChanged(Location location) {
            if (location != null) {
                Log.d(TAG, String.format("%f, %f", location.getLatitude(), location.getLongitude()));
                drawMaker(location);
                locationManager.removeUpdates(locationListener);
            } else {
                Log.d(TAG, "Loaction is null");
            }

        }


        @Override
        public void onStatusChanged(String provider, int status, Bundle extras) {

        }

        @Override
        public void onProviderEnabled(String provider) {

        }

        @Override
        public void onProviderDisabled(String provider) {

        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);

        edtDestination = (EditText) findViewById(R.id.edtDesti);
        imgClear = (ImageView) findViewById(R.id.imgClearDesti);

        imgClear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                edtDestination.setText("");
            }
        });

       mapFragment = (MapFragment) getFragmentManager().findFragmentById(R.id.mapFragment);

        mapFragment.getMapAsync(this);


        //  map.setMapType(GoogleMap.MAP_TYPE_SATELLITE);

        locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
        initMap();
        getCurrentLocation();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mapFragment.onDestroy();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mapFragment.onResume();
        getCurrentLocation();
    }

    @Override
    protected void onPause() {
        super.onPause();
        mapFragment.onPause();
        locationManager.removeUpdates(locationListener);
    }


    @Override
    public void onMapReady(GoogleMap googleMap) {
        map = googleMap;
        btnFLoat = (FloatingActionButton) findViewById(R.id.btnActionFloat);
        btnFLoat.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getCurrentLocation();
            }
        });
        // map.setMyLocationEnabled(true);

        initMap();

    }

    private void initMap() {
        GoogleApiAvailability googleApiAvailability = GoogleApiAvailability.getInstance();
        int googlePlayStatus = googleApiAvailability.isGooglePlayServicesAvailable(this);
        if (googlePlayStatus != ConnectionResult.SUCCESS) {
            googleApiAvailability.getErrorDialog(this, googlePlayStatus, -1).show();
            finish();
        } else {
            if (map != null) {
                if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                    // TODO: Consider calling
                    //    ActivityCompat#requestPermissions
                    // here to request the missing permissions, and then overriding
                    //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                    //                                          int[] grantResults)
                    // to handle the case where the user grants the permission. See the documentation
                    // for ActivityCompat#requestPermissions for more details.
                    return;
                }
                map.setMyLocationEnabled(true);
                map.getUiSettings().setMyLocationButtonEnabled(true);
                map.getUiSettings().setAllGesturesEnabled(true);
            }
        }
    }

    private void getCurrentLocation() {
        boolean isGPSEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);
        boolean isNetworkEnabled = locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER);

        Location location = null;
        if(!(isGPSEnabled || isNetworkEnabled)) {
            Toast.makeText(this, R.string.error_location_provider,Toast.LENGTH_LONG).show();
        }
        else {
                if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                    // TODO: Consider calling
                    //    ActivityCompat#requestPermissions
                    // here to request the missing permissions, and then overriding
                    //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                    //                                          int[] grantResults)
                    // to handle the case where the user grants the permission. See the documentation
                    // for ActivityCompat#requestPermissions for more details.
                    return;
                }
                locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER,LOCATION_UPDATE_MIN_TIME,
                        LOCATION_UPDATE_MIN_DISTANCE, locationListener);
                location = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
                location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);

        }

        if(location != null) {
            Log.d(TAG, String.format("getCurrentLocation(%f,%f)", location.getLatitude(), location.getLongitude()));
            drawMaker(location);
        }
    }

    private void drawMaker(Location location) {
        if(map != null) {
            map.clear();
            LatLng gps = new LatLng(location.getLatitude(), location.getLongitude());
            map.addMarker(new MarkerOptions().position(gps).title("Current position"));
            map.animateCamera(CameraUpdateFactory.newLatLngZoom(gps, 17));
        }
    }


}
