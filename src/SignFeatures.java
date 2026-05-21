import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

import java.io.File;

/**
 * Descripteurs HOG sur le panneau entier (niveaux de gris, 64×64).
 */
public class SignFeatures {

    public static final int SIGN_SIZE = 96;

    public static Mat toHog(File imageFile) {
        Mat image = Imgcodecs.imread(imageFile.getAbsolutePath());
        if (image.empty()) {
            return new Mat();
        }
        return toHogFromMat(image);
    }

    public static Mat toHogFromMat(Mat image) {
        if (image == null || image.empty()) {
            return new Mat();
        }

        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(SIGN_SIZE, SIGN_SIZE));

        Mat gray = new Mat();
        Imgproc.cvtColor(resized, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(gray, gray);

        HOGDescriptor hog = new HOGDescriptor(
                new Size(SIGN_SIZE, SIGN_SIZE),
                new Size(16, 16),
                new Size(8, 8),
                new Size(8, 8),
                9
        );

        MatOfFloat descriptors = new MatOfFloat();
        hog.compute(gray, descriptors);

        float[] values = descriptors.toArray();
        Mat feature = new Mat(1, values.length, CvType.CV_32F);
        feature.put(0, 0, values);
        return feature;
    }
}
