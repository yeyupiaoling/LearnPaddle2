package com.yeyupiaoling.note15;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.baidu.paddle.PML;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private String model_path;
    private String assets_path = "infer_model";
    private boolean load_result = false;
    private int[] ddims = {1, 3, 224, 224};
    private ImageView imageView;
    private TextView showTv;

    static {
        try {
            System.loadLibrary("paddle-mobile");

        } catch (Exception e) {
            e.printStackTrace();

        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initView();
        requestPermissions();
        Utils.copyFileFromAsset(this, assets_path, model_path);
    }

    private void initView(){
        Button loadBtn = findViewById(R.id.load);
        Button clearBtn = findViewById(R.id.clear);
        Button inferBtn = findViewById(R.id.infer);
        showTv = findViewById(R.id.show);
        imageView = findViewById(R.id.image_view);

        model_path = getCacheDir().getAbsolutePath() + File.separator + "infer_model";


        loadBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                load_result = PML.load(model_path);
                if (load_result) {
                    Toast.makeText(MainActivity.this, "模型加载成功", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(MainActivity.this, "模型加载失败", Toast.LENGTH_SHORT).show();
                }
            }
        });

        clearBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                PML.clear();
                load_result = false;
                Toast.makeText(MainActivity.this, "模型已清空", Toast.LENGTH_SHORT).show();
            }
        });

        inferBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (load_result){
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("image/*");
                    startActivityForResult(intent, 1);
                } else {
                    Toast.makeText(MainActivity.this, "模型未加载", Toast.LENGTH_SHORT).show();
                }
            }
        });

    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        String image_path;
        if (resultCode == Activity.RESULT_OK) {
            switch (requestCode) {
                case 1:
                    if (data == null) {
                        return;
                    }
                    Uri image_uri = data.getData();
                    // get image path from uri
                    image_path = Utils.getPathFromURI(MainActivity.this, image_uri);
                    Bitmap bitmap = Utils.getScaleBitmap(image_path);
                    imageView.setImageBitmap(bitmap);
                    // predict image
                    predictImage(image_path);
                    break;
            }
        }
    }


    private void predictImage(String image_path) {
        // picture to float array
        Bitmap bmp = Utils.getScaleBitmap(image_path);
        float[] inputData = Utils.getScaledMatrix(bmp, ddims[2], ddims[3]);
        try {
            long start = System.currentTimeMillis();
            // get predict result
            float[] result = PML.predictImage(inputData, ddims);
            long end = System.currentTimeMillis();
            // show predict result and time
            int r = Utils.getMaxResult(result);
            String[] names = {"苹果", "哈密瓜", "胡萝卜", "樱桃", "黄瓜", "西瓜"};
            String show_text = "标签：" + r + "\n名称：" + names[r] + "\n概率：" + result[r] + "\n时间：" + (end - start) + "ms";
            showTv.setText(show_text);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void requestPermissions() {

        List<String> permissionList = new ArrayList<>();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.READ_EXTERNAL_STORAGE);
        }

        // if list is not empty will request permissions
        if (!permissionList.isEmpty()) {
            ActivityCompat.requestPermissions(this, permissionList.toArray(new String[permissionList.size()]), 1);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case 1:
                if (grantResults.length > 0) {
                    for (int i = 0; i < grantResults.length; i++) {

                        int grantResult = grantResults[i];
                        if (grantResult == PackageManager.PERMISSION_DENIED) {
                            String s = permissions[i];
                            Toast.makeText(this, s + " permission was denied", Toast.LENGTH_SHORT).show();
                        }
                    }
                }
                break;
        }
    }
}
