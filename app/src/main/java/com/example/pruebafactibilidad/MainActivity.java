package com.example.pruebafactibilidad;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import android.graphics.YuvImage;
import android.graphics.Rect;
import java.io.ByteArrayOutputStream;
import android.content.res.AssetFileDescriptor;
import android.graphics.ImageFormat;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.ExecutionException;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;
import android.graphics.Matrix;

import com.google.common.util.concurrent.ListenableFuture;

public class MainActivity extends AppCompatActivity {
    private PreviewView previewView;
    private ImageAnalysis imageAnalysis;
    private Interpreter tflite;
    private ImageReader imageReader;
    private final int modelInputSize = 128; // Ajusta según tu modelo

    // Asumiendo que tienes un TextView en tu layout con el id "resultTextView"
    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

       resultTextView = findViewById(R.id.resultTextView);

        initializeTFLite();
        initializeCamera();
    }

    private final String[] CLASSES = {"clase1", "clase2", "clase3", "clase4", "clase5", "clase6"};

    private void processBitmap(Bitmap bitmap) {

        // Verifica si el bitmap es nulo
        if (bitmap == null) {
            // Log para debugging
            Log.e("MainActivity", "Received a null bitmap!");
            return; // Termina la ejecución del método aquí y regresa
        }
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, modelInputSize, modelInputSize, true);
        ByteBuffer imgData = convertBitmapToByteBuffer(resizedBitmap);

        float[][] output = new float[1][6]; // Ajusta según tu número de clases
        tflite.run(imgData, output);

        // Obtener el índice de la clase con el valor más alto
        int predictedClass = argMax(output[0]);
        String className = CLASSES[predictedClass];

        // Mostrar el nombre de la clase en el TextView
        resultTextView.setText(className);
    }

    // Este método devuelve el índice del valor más alto en un arreglo
    private int argMax(float[] elements) {
        int maxIndex = -1;
        float maxValue = Float.MIN_VALUE;
        for (int i = 0; i < elements.length; i++) {
            if (elements[i] > maxValue) {
                maxValue = elements[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }


    private void initializeTFLite() {
        try {
            // Cargar modelo
            Interpreter.Options tfliteOptions = new Interpreter.Options();
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("MiNeurona.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void initializeCamera() {
        previewView = findViewById(R.id.previewView); // Asegúrate de tener una PreviewView en tu layout

        imageAnalysis = new ImageAnalysis.Builder().setTargetResolution(new Size(modelInputSize, modelInputSize))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                // Convertir ImageProxy a Bitmap
                Bitmap bitmap = toBitmap(image);
                processBitmap(bitmap); // Suponiendo que tienes un método processBitmap para procesar y realizar inferencia
                image.close();
            }
        });

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                cameraProvider.bindToLifecycle(MainActivity.this, cameraSelector, preview, imageAnalysis);

            } catch (ExecutionException | InterruptedException e) {
                // Manejar excepción
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private final ImageReader.OnImageAvailableListener onImageAvailableListener = new ImageReader.OnImageAvailableListener() {
        @Override
        public void onImageAvailable(ImageReader reader) {
            try (Image image = reader.acquireLatestImage()) {
                ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                byte[] bytes = new byte[buffer.remaining()];
                buffer.get(bytes);

                // Convierte los bytes en un Bitmap
                Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
                processBitmap(bitmap);
            }
        }
    };
    private Bitmap toBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);
        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer imgData = ByteBuffer.allocateDirect(4 * modelInputSize * modelInputSize * 3);
        imgData.order(ByteOrder.nativeOrder());

        for (int y = 0; y < modelInputSize; y++) {
            for (int x = 0; x < modelInputSize; x++) {
                int pixelValue = bitmap.getPixel(x, y);
                imgData.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f);
                imgData.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);
                imgData.putFloat((pixelValue & 0xFF) / 255.0f);
            }
        }

        return imgData;
    }
}