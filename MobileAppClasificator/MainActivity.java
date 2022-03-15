package com.example.melanomaclassification;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.animation.ValueAnimator;
import android.annotation.TargetApi;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;

import com.example.melanomaclassification.ml.ConvertedModel;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    TextView resultText, confidenceText, result, confidence;
    ImageView imageView;
    ImageButton picture,upload_pic,card1,card2;
    int imageSize = 384;
    static final int REQUEST_IMAGE_CAPTURE = 0;
    static final int IMAGE_PICK_CODE = 1000;
    static final int PERMISSION_CODE = 1001;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        resultText = findViewById(R.id.classified);
        confidence = findViewById(R.id.confidence);
        confidenceText = findViewById(R.id.confidencesText);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        upload_pic = findViewById(R.id.button_for_galery);
        card1 = findViewById(R.id.button_for_card);
        card2 = findViewById(R.id.button_for_card_2);
        card1.setVisibility(View.GONE);
        card2.setVisibility(View.GONE);

        upload_pic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                    if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE)== PackageManager.PERMISSION_DENIED){
                    String[] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE};
                    requestPermissions(permissions,PERMISSION_CODE);
                    }
                    else{
                        pickImageFromGallery();
                    }
                }
                else{

                }
            }
        });
        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                takePictureIntent.putExtra("android.intent.extras.CAMERA_FACING",0);
                try {
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
                } catch (ActivityNotFoundException e) {
                    // display error state to the user
                }

            }
        });
    }
    private void pickImageFromGallery(){
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, IMAGE_PICK_CODE);
    }
    @Override
    public void onBackPressed() {
        Intent intent = new Intent(this, MainActivity.class);
        startActivity(intent);
        finish();
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode){
            case PERMISSION_CODE:{
                if (grantResults.length >0 && grantResults[0] == PackageManager.PERMISSION_GRANTED){
                    pickImageFromGallery();
                }
                else{
                    Toast.makeText(this,"Permission denied..!",Toast.LENGTH_SHORT).show();
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    public void classifyImage(Bitmap image){
        try {
            ConvertedModel model = ConvertedModel.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 384, 384, 3}, DataType.FLOAT32);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // get 1D array of 224 * 224 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            float maxConfidence = 0;
            int maxPos;
            String[] classes = {"Бенигна", "Меланома (Малигна)"};
            float s = 0;
            if (confidences[0] >= 0.5) {
                maxPos = 1;
                s = confidences[0] * 100;
            }
            else{
                maxPos = 0;
                s = 100-(confidences[0] * 100);
            }


            float scoreS = BigDecimal.valueOf(s)
                    .setScale(2, BigDecimal.ROUND_HALF_DOWN)
                    .floatValue();

            card1.setVisibility(View.VISIBLE);
            card2.setVisibility(View.VISIBLE);
            resultText.setText("Кожната лезија е:");
            result.setText(classes[maxPos]);
            confidenceText.setText("Ниво на доверба:");
            final ValueAnimator anim = ValueAnimator.ofInt(0,Math.round(scoreS));
            anim.setDuration(1900);
            anim.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
                @Override
                public void onAnimationUpdate(ValueAnimator valueAnimator) {
                    confidence.setText(String.format("%s%%\n", anim.getAnimatedValue().toString()));
                }
            });
            anim.start();

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }
    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap image = (Bitmap) extras.get("data");

            int dimension = Math.min(image.getWidth(),image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
            classifyImage(image);
        }
        if (resultCode == RESULT_OK && requestCode == IMAGE_PICK_CODE){
            Uri imageUri = data.getData();
            try {
                InputStream stream = getContentResolver().openInputStream(imageUri);
                Bitmap image = BitmapFactory.decodeStream(stream);
                int dimension = Math.min(image.getWidth(),image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
                classifyImage(image);
            }
            catch(Exception e){
                e.printStackTrace();
            }

        }
        super.onActivityResult(requestCode, resultCode, data);
    }


}