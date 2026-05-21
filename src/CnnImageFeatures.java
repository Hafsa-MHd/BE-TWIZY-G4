import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Blob NCHW [1, 3, 64, 64] — même format que l'export ONNX (channels_first).
 */
public class CnnImageFeatures {

    public static final int CNN_SIZE = 64;

    public static Mat blobFromSign(Mat signBgr) {
        if (signBgr == null || signBgr.empty()) {
            return new Mat();
        }

        Mat rgb = new Mat();
        Imgproc.cvtColor(signBgr, rgb, Imgproc.COLOR_BGR2RGB);
        Mat resized = new Mat();
        Imgproc.resize(rgb, resized, new Size(CNN_SIZE, CNN_SIZE));

        int[] dims = {1, 3, CNN_SIZE, CNN_SIZE};
        Mat blob = new Mat(dims, CvType.CV_32F);
        float[] data = new float[3 * CNN_SIZE * CNN_SIZE];

        for (int y = 0; y < CNN_SIZE; y++) {
            for (int x = 0; x < CNN_SIZE; x++) {
                double[] px = resized.get(y, x);
                int o = y * CNN_SIZE + x;
                data[0 * CNN_SIZE * CNN_SIZE + o] = (float) (px[0] / 255.0);
                data[1 * CNN_SIZE * CNN_SIZE + o] = (float) (px[1] / 255.0);
                data[2 * CNN_SIZE * CNN_SIZE + o] = (float) (px[2] / 255.0);
            }
        }

        blob.put(0, 0, data);
        return blob;
    }
}
