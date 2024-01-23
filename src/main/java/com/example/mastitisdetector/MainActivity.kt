package com.example.mastitisdetector

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.Nullable
import androidx.appcompat.app.AppCompatActivity
import com.example.mastitisdetector.ml.MD
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {


    private lateinit var imgView: ImageView
    private lateinit var select: Button
    private lateinit var predict: Button
    private lateinit var tv: TextView
    private lateinit var img: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imgView = findViewById(R.id.imageView)
        tv = findViewById(R.id.textView)
        select = findViewById(R.id.button)
        predict = findViewById(R.id.button2)



        select.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }

        predict.setOnClickListener {
            val model = MD.newInstance(this)

            // Resize the image
            val resizedImage = Bitmap.createScaledBitmap(img, 256, 256, true)

            // Create input tensor from the resized image
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(resizedImage)

            // Create input buffer with correct shape
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)

            // Load the tensor image data into the input buffer
            inputFeature0.loadBuffer(tensorImage.buffer)

            // Run model inference and get the result
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            // Release model resources if no longer used
            model.close()

            // Display the prediction result in the TextView
            val prediction = outputFeature0.floatArray[0] // Assuming a single prediction value
            // Check the value for possibility of disease
            val result = if (prediction > 0.5) {
                "Predicted Normal"
            } else {
                "Predicted Mastitis"
            }

            // Display the prediction result in the TextView
            tv.text = "Prediction: $prediction\n$result"

        }

    }






    override fun onActivityResult(requestCode: Int, resultCode: Int, @Nullable data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 100 && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                imgView.setImageURI(uri)

                try {
                    img = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
        }
    }
}