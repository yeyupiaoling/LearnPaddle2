
# 目录
@[toc]
# 前言
现在越来越多的手机要使用到深度学习了，比如一些图像分类，目标检测，风格迁移等等，之前都是把数据提交给服务器完成的。但是提交给服务器有几点不好，首先是速度问题，图片上传到服务器需要时间，客户端接收结果也需要时间，这一来回就占用了一大半的时间，会使得整体的预测速度都变慢了，再且现在手机的性能不断提高，足以做深度学习的预测。其二是隐私问题，如果只是在本地预测，那么用户根本就不用上传图片，安全性也大大提高了。所以本章我们就来学如何包我们训练的PaddlePaddle预测模型部署到Android手机上。


# 编译paddle-mobile库
想要把PaddlePaddle训练好的预测库部署到Android手机上，还需要借助paddle-mobile框架。paddle-mobile框架主要是为了方便PaddlePaddle训练好的模型部署到移动设备上，比如Android手机，苹果手机，树莓派等等这些移动设备，有了paddle-mobile框架大大方便了把PaddlePaddle的预测库部署到移动设备上，而且paddle-mobile框架针对移动设备做了大量的优化，使用这些预测库在移动设备上有了更好的预测性能。

想要在Android手机上使用paddle-mobile，就要编译Android能够使用的CPP库，在这一部分中，我们介绍两种编译Android的paddle-mobile库，分别是使用Docker编译paddle-mobile库、使用Ubuntu交叉编译paddle-mobile库。

## 使用Docker编译

为了方便操作，以下的操作都是在root用户的执行的：

1、安装Docker，以下是在Ubuntu下安装的的方式，只要一条命令就可以了：
```
apt-get install docker.io
```

2、克隆paddle-mobile源码：
```
git clone https://github.com/PaddlePaddle/paddle-mobile.git
```

3、进入到paddle-mobile根目录下编译docker镜像：
```
cd paddle-mobile
# 编译生成进行，编译时间可能要很长
docker build -t paddle-mobile:dev - < Dockerfile
```

编译完成可以使用`docker images`命令查看是否已经生成进行：
```
root@test:/home/test# docker images
REPOSITORY                          TAG                 IMAGE ID            CREATED             SIZE
paddle-mobile                       dev                 fffbd8779c68        20 hours ago        3.76 GB
```

4、运行镜像并进入到容器里面，当前目录还是在paddle-mobile根目录下：
```
docker run -it -v $PWD:/paddle-mobile paddle-mobile:dev
```

5、在容器里面执行以下两条命令：
```
root@fc6f7e9ebdf1:/# cd paddle-mobile/
root@fc6f7e9ebdf1:/paddle-mobile# cmake -DCMAKE_TOOLCHAIN_FILE=tools/toolchains/arm-android-neon.cmake
```

6、（可选）可以使用命令`ccmake .`配置一些信息，比如可以设置`NET`仅支持`googlenet`，这样便于得到的paddle-mobile库会更小一些，修改完成之后，使用`c`命令保存，使用`g`退出。笔者一般跳过这个步骤。
```
                                                    Page 1 of 1
 CMAKE_ASM_FLAGS                                                                                                                                                                                
 CMAKE_ASM_FLAGS_DEBUG                                                                                                                                                                          
 CMAKE_ASM_FLAGS_RELEASE                                                                                                                                                                        
 CMAKE_BUILD_TYPE                                                                                                                                                                               
 CMAKE_INSTALL_PREFIX             /usr/local                                                                                                                                                    
 CMAKE_TOOLCHAIN_FILE             /paddle-mobile/tools/toolchains/arm-android-neon.cmake                                                                                                        
 CPU                              ON                                                                                                                                                            
 DEBUGING                         ON                                                                                                                                                            
 FPGA                             OFF                                                                                                                                                           
 LOG_PROFILE                      ON                                                                                                                                                            
 MALI_GPU                         OFF                                                                                                                                                           
 NET                              defult                                                                                                                                                        
 USE_EXCEPTION                    ON                                                                                                                                                            
 USE_OPENMP                       ON                                                       
```

7、最后执行一下`make`就可以了，到这一步就完成了paddle-mobile的编译。
```
root@fc6f7e9ebdf1:/paddle-mobile# make
```

8、使用`exit`命令退出容器，回到Ubuntu本地上。
```
root@fc6f7e9ebdf1:/paddle-mobile# exit
```

9、在paddle-mobile根目录下，有一个build目录，我们编译好的paddle-mobile库就在这里。
```
root@test:/home/test/paddle-mobile/build# ls
libpaddle-mobile.so
```
`libpaddle-mobile.so`就是我们在开发Android项目的时候使用到的paddle-mobile库。


## 使用Ubuntu编译

1、首先要下载和解压NDK。
```
wget https://dl.google.com/android/repository/android-ndk-r17b-linux-x86_64.zip
unzip android-ndk-r17b-linux-x86_64.zip
```

2、设置NDK环境变量，目录是NDK的解压目录。
```
export NDK_ROOT="/home/test/paddlepaddle/android-ndk-r17b"
```

设置好之后，可以使用以下的命令查看配置情况。
```
root@test:/home/test/paddlepaddle# echo $NDK_ROOT
/home/test/paddlepaddle/android-ndk-r17b
```

3、安装cmake，需要安装较高版本的，笔者的cmake版本是3.11.2。

 - 下载cmake源码
```
wget https://cmake.org/files/v3.11/cmake-3.11.2.tar.gz
```

 - 解压cmake源码
```
tar -zxvf cmake-3.11.2.tar.gz
```

 - 进入到cmake源码根目录，并执行`bootstrap`。
```
cd cmake-3.11.2
./bootstrap
```

 - 最后执行以下两条命令开始安装cmake。
```
make
make install
```

 - 安装完成之后，可以使用`cmake --version`是否安装成功.
```
root@test:/home/test/paddlepaddle# cmake --version
cmake version 3.11.2

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

4、克隆paddle-mobile源码。
```
git clone https://github.com/PaddlePaddle/paddle-mobile.git
```

5、进入到paddle-mobile的tools目录下，执行编译。
```
cd paddle-mobile/tools/
sh build.sh android
```

（可选）如果想编译针对某一个网络编译更小的库时，可以在命令后面加上相应的参数，如下：
```
sh build.sh android mobilenet
```

6、最后会在`paddle-mobile/build/release/arm-v7a/build`目录下生产paddle-mobile库。
```
root@test:/home/test/paddlepaddle/paddle-mobile/build/release/arm-v7a/build# ls
libpaddle-mobile.so
```
`libpaddle-mobile.so`就是我们在开发Android项目的时候使用到的paddle-mobile库。


# 创建Android项目

首先使用Android Studio创建一个普通的Android项目，我们可以不用选择CPP的支持，因为我们已经编译好了CPP。之后按照以下的步骤开始执行：

1、在`main`目录下创建两个`assets/infer_model`文件夹，这个文件夹我们将会使用它来存放PaddlePaddle训练好的预测模型，本章我们使用的预测模型是[《PaddlePaddle从入门到炼丹》十一——自定义图像数据集识别](https://blog.csdn.net/qq_33200967/article/details/87895105)训练得到的预测模型，我们训练好的模型复制到这个文件夹下。

2、在`main`目录下创建一个`jniLibs`文件夹，这个文件夹是存放CPP编译库的，就是**编译paddle-mobile库**部分编译的`libpaddle-mobile.so`

3、在Android项目的配置文件夹中加上权限声明，因为我们要使用到读取相册和使用相机，所以加上以下的权限声明：
```xml
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

4、修改`activity_main.xml`界面，修改成如下：
```xml
<?xml version="1.0" encoding="utf-8"?>
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".MainActivity">

    <LinearLayout
        android:id="@+id/ll"
        android:orientation="horizontal"
        android:layout_alignParentBottom="true"
        android:layout_width="match_parent"
        android:layout_height="50dp">

        <Button
            android:layout_weight="1"
            android:id="@+id/load"
            android:text="加载模型"
            android:layout_width="0dp"
            android:layout_height="match_parent" />

        <Button
            android:id="@+id/clear"
            android:layout_weight="1"
            android:text="清空模型"
            android:layout_width="0dp"
            android:layout_height="match_parent" />

        <Button
            android:id="@+id/infer"
            android:layout_weight="1"
            android:text="预测图片"
            android:layout_width="0dp"
            android:layout_height="match_parent" />
    </LinearLayout>

    <TextView
        android:layout_above="@id/ll"
        android:id="@+id/show"
        android:hint="这里显示预测结果"
        android:layout_width="match_parent"
        android:layout_height="100dp" />

    <ImageView
        android:id="@+id/image_view"
        android:layout_above="@id/show"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />
</RelativeLayout>
```

5、创建一个`com.baidu.paddle`包，在这个包下创建的Java程序，这个Java程序就是用于调用paddle-mobile的CPP动态库的。它提供了多种方法给我们使用，我们主要使用到加载模型的方法`load(String modelDir)`，清空已加载的方法`clear()`，还有最最重要的预测方法`predictImage(float[] buf, int[]ddims)`。
```java
package com.baidu.paddle;

public class PML {
    // set thread num
    public static native void setThread(int threadCount);

    //Load seperated parameters
    public static native boolean load(String modelDir);

    // load qualified model
    public static native boolean loadQualified(String modelDir);

    // Load combined parameters
    public static native boolean loadCombined(String modelPath, String paramPath);

    // load qualified model
    public static native boolean loadCombinedQualified(String modelPath, String paramPath);

    // object detection
    public static native float[] predictImage(float[] buf, int[]ddims);

    // predict yuv image
    public static native float[] predictYuv(byte[] buf, int imgWidth, int imgHeight, int[] ddims, float[]meanValues);

    // clear model
    public static native void clear();
}
```

6、然后在项目的主要包下创建一个`Utils.java`的工具类。这个工具类主要编写一些图像的处理方法，和一些模型复制方法等，我们下面将一一介绍这些方法。

该方法是用于获取预测结果中概率最大的标签，参数是执行预测的结果，这个结果是对应没有类别的概率，这个方法就判断哪个类别的概率最大，然后就返回概率最大的标签。
```java
// 获取预测值中最大概率的标签
public static int getMaxResult(float[] result) {
    float probability = result[0];
    int r = 0;
    for (int i = 0; i < result.length; i++) {
        if (probability < result[i]) {
            probability = result[i];
            r = i;
        }
    }
    return r;
}
```

该方法是把图片转换成预测需要用的数据格式浮点数组。在转换的过程中也对图像做了预处理，这个预处理需要跟训练的预处理的方式一样，否则无法正确预测。还有指定了处理后图片的大小，根据参数输入的宽度和高度，把图片压缩到这些自定的大小。还有把图片的通道顺序改为RGB，同时每个像素除以255，这个操作跟训练的时候一样。
```java
// 对将要预测的图片进行预处理
public static float[] getScaledMatrix(Bitmap bitmap, int desWidth, int desHeight) {
    float[] dataBuf = new float[3 * desWidth * desHeight];
    int rIndex;
    int gIndex;
    int bIndex;
    int[] pixels = new int[desWidth * desHeight];
    Bitmap bm = Bitmap.createScaledBitmap(bitmap, desWidth, desHeight, false);
    bm.getPixels(pixels, 0, desWidth, 0, 0, desWidth, desHeight);
    int j = 0;
    int k = 0;
    for (int i = 0; i < pixels.length; i++) {
        int clr = pixels[i];
        j = i / desHeight;
        k = i % desWidth;
        rIndex = j * desWidth + k;
        gIndex = rIndex + desHeight * desWidth;
        bIndex = gIndex + desHeight * desWidth;
        // 转成RGB通道顺序，并除以255，跟训练的预处理一样
        dataBuf[rIndex] = (float) (((clr & 0x00ff0000) >> 16) / 255.0);
        dataBuf[gIndex] = (float) (((clr & 0x0000ff00) >> 8) / 255.0);
        dataBuf[bIndex] = (float) (((clr & 0x000000ff)) / 255.0);

    }
    if (bm.isRecycled()) {
        bm.recycle();
    }
    return dataBuf;
}
```

该方法是对图片进行压缩，避免图片过大，超过内存支出。把图片的最大长度压缩到500以内。
```java
// 压缩图片，避免图片过大
public static Bitmap getScaleBitmap(String filePath) {
    BitmapFactory.Options opt = new BitmapFactory.Options();
    opt.inJustDecodeBounds = true;
    BitmapFactory.decodeFile(filePath, opt);

    int bmpWidth = opt.outWidth;
    int bmpHeight = opt.outHeight;

    int maxSize = 500;

    // compress picture with inSampleSize
    opt.inSampleSize = 1;
    while (true) {
        if (bmpWidth / opt.inSampleSize < maxSize || bmpHeight / opt.inSampleSize < maxSize) {
            break;
        }
        opt.inSampleSize *= 2;
    }
    opt.inJustDecodeBounds = false;
    return BitmapFactory.decodeFile(filePath, opt);
}
```

该方法是根据相册返回的URI转换为图片的绝对路径，用于之后使用这个路径获取图片内容。
```java
// 根据相册返回的URI返回图片的绝对路径
public static String getPathFromURI(Context context, Uri uri) {
    String result;
    Cursor cursor = context.getContentResolver().query(uri, null, null, null, null);
    if (cursor == null) {
        result = uri.getPath();
    } else {
        cursor.moveToFirst();
        int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
        result = cursor.getString(idx);
        cursor.close();
    }
    return result;
}
```

该方法是把`assets`资源文件下的预测文件复制到缓存目录，用于之后加载模型文件。
```java
// 复制莫模型文件到缓存目录
public static void copyFileFromAsset(Context context, String oldPath, String newPath) {
    try {
        // 预测模型文件在assets中的位置
        String[] fileNames = context.getAssets().list(oldPath);
        if (fileNames.length > 0) {
            // directory
            File file = new File(newPath);
            if (!file.exists()) {
                file.mkdirs();
            }
            // copy recursivelyC
            for (String fileName : fileNames) {
                copyFileFromAsset(context, oldPath + "/" + fileName, newPath + "/" + fileName);
            }
        } else {
            // file
            File file = new File(newPath);
            // if file exists will never copy
            if (file.exists()) {
                return;
            }

            // copy file to new path
            InputStream is = context.getAssets().open(oldPath);
            FileOutputStream fos = new FileOutputStream(file);
            byte[] buffer = new byte[1024];
            int byteCount;
            while ((byteCount = is.read(buffer)) != -1) {
                fos.write(buffer, 0, byteCount);
            }
            fos.flush();
            is.close();
            fos.close();
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
}
```


7、最后修改`MainActivity.java`，修改如下：

这里做一些初始化操作，如加载PaddleMobile的动态库，指定图片的形状。
```java
    private String model_path;
    // 模型文件夹
    private String assets_path = "infer_model";
    private boolean load_result = false;
    // 输入图片的形状，分别是：batch size、通道数、宽度、高度
    private int[] ddims = {1, 3, 224, 224};
    private ImageView imageView;
    private TextView showTv;

    // 加载PaddleMobile的动态库
    static {
        try {
            System.loadLibrary("paddle-mobile");

        } catch (Exception e) {
            e.printStackTrace();

        }
    }
```

该方法是初始化控件，和定义按钮的点击事件，如加载模型点击事件，清空模型点击事件，打开相册预测图片点击事件。
```java
    // 初始化控件
    private void initView(){
        Button loadBtn = findViewById(R.id.load);
        Button clearBtn = findViewById(R.id.clear);
        Button inferBtn = findViewById(R.id.infer);
        showTv = findViewById(R.id.show);
        imageView = findViewById(R.id.image_view);

        // 加载模型点击事件
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

        // 清空模型点击事件
        clearBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                PML.clear();
                load_result = false;
                Toast.makeText(MainActivity.this, "模型已清空", Toast.LENGTH_SHORT).show();
            }
        });

        // 打开相册选择图片预测点击事件
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
```

该方法是一个回调方法，主要是打开相册后的回调预测操作。使用返回的URI转换为绝对路径，然后使用这个图片路径转换成Bitmap用于显示，同时也使用这个路径执行预测操作。
```java
    // 回调事件
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        String image_path;
        if (resultCode == Activity.RESULT_OK) {
            switch (requestCode) {
                case 1:
                    if (data == null) {
                        return;
                    }
                    // 获取相册返回的URI
                    Uri image_uri = data.getData();
                    // 根据图片的URI获取绝对路径
                    image_path = Utils.getPathFromURI(MainActivity.this, image_uri);
                    // 压缩图片用于显示
                    Bitmap bitmap = Utils.getScaleBitmap(image_path);
                    imageView.setImageBitmap(bitmap);
                    // 开始预测图片
                    predictImage(image_path);
                    break;
            }
        }
    }
```


该方法是预测操作的方法，参数是图片的绝对路径，首先根据图片获取已经压缩过的Bitmap，然后使用这个Bitmap转换成预处理后的浮点数组，最后执行预测操作。再根据预测结果提取最大概率的标签，并获取该标签的类别名称。
```java
    // 根据图片的路径预测图片
    private void predictImage(String image_path) {
        // 把图片进行压缩
        Bitmap bmp = Utils.getScaleBitmap(image_path);
        // 把图片转换成浮点数组，用于预测
        float[] inputData = Utils.getScaledMatrix(bmp, ddims[2], ddims[3]);
        try {
            long start = System.currentTimeMillis();
            // 执行预测，获取预测结果
            float[] result = PML.predictImage(inputData, ddims);
            long end = System.currentTimeMillis();
            // 获取概率最大的标签
            int r = Utils.getMaxResult(result);
            // 获取标签对应的类别名称
            String[] names = {"苹果", "哈密瓜", "胡萝卜", "樱桃", "黄瓜", "西瓜"};
            String show_text = "标签：" + r + "\n名称：" + names[r] + "\n概率：" + result[r] + "\n时间：" + (end - start) + "ms";
            // 显示预测结果
            showTv.setText(show_text);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
```


这主要是用于动态获取权限，因为读取外部文件需要读取外部文件的权限，又因为读取外部文件权限是属于危险权限，需要动态获取。
```java
    // 多权限动态申请
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
```

然后修改`onCreate`，首先获取缓存文件路径，然后初始化视图控件和动态获取权限，最后把预测模型文件复制到缓存路径下。
```java
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        model_path = getCacheDir().getAbsolutePath() + File.separator + "infer_model";
        // 初始化控件
        initView();
        // 动态请求权限
        requestPermissions();
        // 从assets中复制模型文件到缓存目录下
        Utils.copyFileFromAsset(this, assets_path, model_path);
    }
```


8、最后运行项目，选择图片预测会得到以下的效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190223192610299.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

# 参考资料

 1. https://github.com/PaddlePaddle/paddle-mobile
 2. https://blog.csdn.net/qq_33200967/article/details/81066970


