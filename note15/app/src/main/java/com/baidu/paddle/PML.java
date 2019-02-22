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
