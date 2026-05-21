import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

import java.io.File;

/**
 * Descripteurs pour panneaux de limitation de vitesse :
 * HOG sur les chiffres noirs + HOG sur la zone centrale en niveaux de gris.
 */
public class SpeedLimitFeatures {

    public static final int BASE_SIZE = 128;
    public static final int PATCH_SIZE = 64;

    private static final Rect CENTER_ROI = new Rect(22, 28, 84, 68);

    public static Mat toHog(File imageFile) {
        Mat image = Imgcodecs.imread(imageFile.getAbsolutePath());
        if (image.empty()) {
            return new Mat();
        }
        return toHogFromMat(image);
    }

    public static Mat toHogFromMat(Mat image) {
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(BASE_SIZE, BASE_SIZE));

        Mat digits = extractBlackDigits(resized);
        Mat grayCenter = extractGrayCenter(resized);

        if (digits.empty() || grayCenter.empty()) {
            return new Mat();
        }

        float[] digitHog = computeHog(digits);
        float[] grayHog = computeHog(grayCenter);
        float[] combined = concat(digitHog, grayHog);

        Mat feature = new Mat(1, combined.length, CvType.CV_32F);
        feature.put(0, 0, combined);
        return feature;
    }

    /** Masque des chiffres à la résolution du panneau (pour analyse de forme). */
    public static Mat extractDigitMask(Mat signImage) {
        Mat resized = new Mat();
        Imgproc.resize(signImage, resized, new Size(BASE_SIZE, BASE_SIZE));
        return buildDigitMask(new Mat(resized, CENTER_ROI));
    }

    public static Mat extractBlackDigits(Mat resized128) {
        Mat blackMask = buildDigitMask(new Mat(resized128, CENTER_ROI));
        Mat result = new Mat();
        Imgproc.resize(blackMask, result, new Size(PATCH_SIZE, PATCH_SIZE));
        return result;
    }

    private static Mat buildDigitMask(Mat center) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(center, hsv, Imgproc.COLOR_BGR2HSV);

        Mat blackMask = new Mat();
        Core.inRange(hsv, new Scalar(0, 0, 0), new Scalar(180, 150, 140), blackMask);

        Mat gray = new Mat();
        Imgproc.cvtColor(center, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);

        Mat otsuMask = new Mat();
        Imgproc.threshold(gray, otsuMask, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        Core.bitwise_or(blackMask, otsuMask, blackMask);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
        Imgproc.morphologyEx(blackMask, blackMask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(blackMask, blackMask, Imgproc.MORPH_CLOSE, kernel);

        // Sépare un peu les chiffres fusionnés (110 sur photo réelle)
        Mat sep = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 3));
        Imgproc.erode(blackMask, blackMask, sep);

        return blackMask;
    }

    private static Mat extractGrayCenter(Mat resized128) {
        Mat center = new Mat(resized128, CENTER_ROI);
        Mat gray = new Mat();
        Imgproc.cvtColor(center, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(gray, gray);

        Mat result = new Mat();
        Imgproc.resize(gray, result, new Size(PATCH_SIZE, PATCH_SIZE));
        return result;
    }

    private static float[] computeHog(Mat patch) {
        HOGDescriptor hog = new HOGDescriptor(
                new Size(PATCH_SIZE, PATCH_SIZE),
                new Size(16, 16),
                new Size(8, 8),
                new Size(8, 8),
                9
        );

        MatOfFloat descriptors = new MatOfFloat();
        hog.compute(patch, descriptors);
        return descriptors.toArray();
    }

    private static float[] concat(float[] a, float[] b) {
        float[] result = new float[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
}
