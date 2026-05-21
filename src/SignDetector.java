import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class SignDetector {

    private static final double MIN_CIRCULARITY = 0.55;
    private static final int MIN_SIGN_PX = 18;

    /** Panneau détecté : crop + position dans l'image (pour vidéo / annotation). */
    public static class Detection {
        private final Mat crop;
        private final Rect bounds;
        private final boolean triangular;

        public Detection(Mat crop, Rect bounds) {
            this(crop, bounds, false);
        }

        public Detection(Mat crop, Rect bounds, boolean triangular) {
            this.crop = crop;
            this.bounds = bounds;
            this.triangular = triangular;
        }

        public Mat getCrop() {
            return crop;
        }

        public Rect getBounds() {
            return bounds;
        }

        public boolean isTriangular() {
            return triangular;
        }
    }

    public static List<Mat> detectCircularRedSigns(Mat image) {
        List<Mat> crops = new ArrayList<Mat>();
        for (Detection d : detectDetailed(image)) {
            crops.add(d.crop);
        }
        return crops;
    }

    public static List<Detection> detectDetailed(Mat image) {
        return detectDetailed(image, false);
    }

    /**
     * @param forVideo true = seuils plus bas, tous les masques rouges + Hough (panneaux lointains).
     */
    public static List<Detection> detectDetailed(Mat image, boolean forVideo) {
        List<Detection> detected = new ArrayList<Detection>();
        List<Rect> usedRects = new ArrayList<Rect>();

        if (image == null || image.empty()) {
            return detected;
        }

        Mat hsv = ImageUtils.bgrToHsv(image);
        double imageArea = image.rows() * image.cols();
        double minContourArea = forVideo
                ? Math.max(40, imageArea * 0.00007)
                : Math.max(80, imageArea * 0.00015);
        int minSignPx = forVideo ? 12 : MIN_SIGN_PX;

        int[][] redPresets = {
                {12, 165, 60},
                {8, 170, 80},
                {6, 170, 110},
                {5, 165, 90}
        };

        for (int[] preset : redPresets) {
            Mat redMask = redThreshold(hsv, preset[0], preset[1], preset[2]);
            tryContours(image, redMask, minContourArea, detected, usedRects, minSignPx);
        }

        tryHoughCircles(image, hsv, detected, usedRects, minSignPx);

        detectTriangularWarningSigns(image, hsv, forVideo, detected, usedRects);

        return detected;
    }

    private static class TriangleCandidate {
        final Detection detection;
        final Rect bounds;
        final double score;

        TriangleCandidate(Detection detection, Rect bounds, double score) {
            this.detection = detection;
            this.bounds = bounds;
            this.score = score;
        }
    }

    /**
     * Panneaux triangulaires : collecte, score, puis au plus 1 triangle sur photo (évite faux positifs).
     */
    private static void detectTriangularWarningSigns(Mat image, Mat hsv, boolean forVideo,
                                                     List<Detection> out, List<Rect> usedRects) {
        if (image == null || image.empty()) {
            return;
        }

        double imageArea = image.rows() * image.cols();
        double minContourArea = forVideo
                ? Math.max(80, imageArea * 0.00010)
                : Math.max(200, imageArea * 0.0012);
        int minSide = forVideo ? 22 : 28;

        ArrayList<TriangleCandidate> candidates = new ArrayList<TriangleCandidate>();

        Mat yellowMask = yellowThreshold(hsv);
        Mat warningMask = warningTriangleThreshold(hsv);
        Mat redMask = redThreshold(hsv, 10, 165, 55);

        collectTrianglesFromMask(image, yellowMask, minContourArea, minSide, imageArea, candidates);
        collectTrianglesFromMask(image, warningMask, minContourArea, minSide, imageArea, candidates);
        collectTrianglesFromRedMask(image, redMask, minContourArea, minSide, imageArea, candidates);

        int maxTriangles = forVideo ? 2 : 1;
        double minScore = forVideo ? 0.48 : 0.52;

        for (TriangleCandidate picked : pickBestTriangles(candidates, maxTriangles, minScore)) {
            if (isDuplicate(usedRects, picked.bounds)) {
                continue;
            }
            out.add(picked.detection);
            usedRects.add(picked.bounds);
            Rect b = picked.bounds;
            Imgproc.rectangle(image, new Point(b.x, b.y),
                    new Point(b.x + b.width, b.y + b.height), new Scalar(255, 200, 0), 2);
        }
    }

    private static void collectTrianglesFromMask(Mat image, Mat mask, double minContourArea, int minSide,
                                                 double imageArea, ArrayList<TriangleCandidate> candidates) {
        for (MatOfPoint contour : findContoursOnMask(mask)) {
            collectTriangleContour(image, contour, minContourArea, minSide, imageArea, candidates);
        }
    }

    private static void collectTrianglesFromRedMask(Mat image, Mat redMask, double minContourArea,
                                                    int minSide, double imageArea,
                                                    ArrayList<TriangleCandidate> candidates) {
        for (MatOfPoint contour : findContoursOnMask(redMask)) {
            double area = Imgproc.contourArea(contour);
            if (area < minContourArea) {
                continue;
            }
            MatOfPoint2f curve = new MatOfPoint2f();
            curve.fromList(contour.toList());
            Point center = new Point();
            float[] radius = new float[1];
            Imgproc.minEnclosingCircle(curve, center, radius);
            if (radius[0] <= 0) {
                continue;
            }
            double circleArea = Math.PI * radius[0] * radius[0];
            if (area / circleArea > 0.58) {
                continue;
            }
            collectTriangleContour(image, contour, minContourArea, minSide, imageArea, candidates);
        }
    }

    private static void collectTriangleContour(Mat image, MatOfPoint contour, double minContourArea,
                                               int minSide, double imageArea,
                                               ArrayList<TriangleCandidate> candidates) {
        int vertices = triangleVertexCount(contour);
        if (!isTriangleContour(contour, minContourArea, vertices)) {
            return;
        }

        Rect rect = Imgproc.boundingRect(contour);
        int pad = Math.max(4, Math.min(rect.width, rect.height) / 8);
        int x = Math.max(rect.x - pad, 0);
        int y = Math.max(rect.y - pad, 0);
        int right = Math.min(x + rect.width + 2 * pad, image.cols());
        int bottom = Math.min(y + rect.height + 2 * pad, image.rows());
        int w = right - x;
        int h = bottom - y;

        if (w < minSide || h < minSide) {
            return;
        }

        double bboxArea = w * (double) h;
        if (bboxArea < imageArea * 0.001 || bboxArea > imageArea * 0.12) {
            return;
        }

        Rect safe = new Rect(x, y, w, h);
        Mat sign = new Mat(image, safe);
        Mat copy = new Mat();
        sign.copyTo(copy);

        if (SignTypeHeuristic.looksLikeSpeedLimitSign(copy)) {
            return;
        }
        if (!SignTypeHeuristic.looksLikeWarningTriangle(copy)) {
            return;
        }

        double score = scoreTriangleCandidate(copy, vertices, bboxArea / imageArea);
        if (score < 0.35) {
            return;
        }

        candidates.add(new TriangleCandidate(new Detection(copy, safe, true), safe, score));
    }

    private static ArrayList<TriangleCandidate> pickBestTriangles(ArrayList<TriangleCandidate> candidates,
                                                                int maxCount, double minScore) {
        ArrayList<TriangleCandidate> sorted = new ArrayList<TriangleCandidate>(candidates);
        sorted.sort((a, b) -> Double.compare(b.score, a.score));

        ArrayList<TriangleCandidate> picked = new ArrayList<TriangleCandidate>();
        for (TriangleCandidate c : sorted) {
            if (c.score < minScore) {
                break;
            }
            boolean overlaps = false;
            for (TriangleCandidate p : picked) {
                if (overlapRatio(p.bounds, c.bounds) > 0.30) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                picked.add(c);
            }
            if (picked.size() >= maxCount) {
                break;
            }
        }
        return picked;
    }

    private static double scoreTriangleCandidate(Mat crop, int vertices, double areaRatio) {
        double score = 0.0;
        if (vertices == 3) {
            score += 0.35;
        } else if (vertices == 4) {
            score += 0.15;
        }
        if (areaRatio >= 0.002 && areaRatio <= 0.06) {
            score += 0.25;
        }
        score += SignTypeHeuristic.warningTriangleColorScore(crop) * 0.40;
        return score;
    }

    private static int triangleVertexCount(MatOfPoint contour) {
        MatOfPoint2f curve = new MatOfPoint2f();
        curve.fromList(contour.toList());
        double peri = Imgproc.arcLength(curve, true);
        if (peri < 1) {
            return 0;
        }
        MatOfPoint2f approx = new MatOfPoint2f();
        Imgproc.approxPolyDP(curve, approx, 0.07 * peri, true);
        return (int) approx.total();
    }

    /** Jaune + blanc brillant (panneaux danger souvent peu saturés sur photo réelle). */
    private static Mat warningTriangleThreshold(Mat hsv) {
        Mat yellow = yellowThreshold(hsv);
        Mat white = new Mat();
        Core.inRange(hsv, new Scalar(0, 0, 160), new Scalar(180, 70, 255), white);
        Mat combined = new Mat();
        Core.bitwise_or(yellow, white, combined);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(combined, combined, Imgproc.MORPH_CLOSE, kernel);
        return combined;
    }

    private static Mat yellowThreshold(Mat hsv) {
        Mat mask = new Mat();
        Core.inRange(hsv, new Scalar(15, 60, 60), new Scalar(40, 255, 255), mask);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
        return mask;
    }

    private static boolean isTriangleContour(MatOfPoint contour, double minArea, int vertices) {
        double area = Imgproc.contourArea(contour);
        if (area < minArea || vertices < 3 || vertices > 4) {
            return false;
        }

        Rect rect = Imgproc.boundingRect(contour);
        if (rect.width < 1 || rect.height < 1) {
            return false;
        }

        double aspect = (double) rect.width / rect.height;
        if (aspect < 0.55 || aspect > 1.8) {
            return false;
        }

        double extent = area / (rect.width * (double) rect.height);
        return extent >= 0.30 && extent <= 0.68;
    }

    private static void tryContours(Mat image, Mat redMask, double minContourArea,
                                    List<Detection> out, List<Rect> usedRects) {
        tryContours(image, redMask, minContourArea, out, usedRects, MIN_SIGN_PX);
    }

    private static void tryContours(Mat image, Mat redMask, double minContourArea,
                                    List<Detection> out, List<Rect> usedRects, int minSignPx) {
        List<MatOfPoint> contours = findContoursOnMask(redMask);

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < minContourArea) {
                continue;
            }

            CropResult crop = extractCleanCircularSign(image, contour);
            if (crop == null || crop.sign.empty()) {
                continue;
            }
            if (crop.sign.cols() < minSignPx || crop.sign.rows() < minSignPx) {
                continue;
            }

            // En vidéo on garde plus de candidats (panneaux lointains / partiellement visibles)
            if (!minSignPxIsVideo(minSignPx) && !SignTypeHeuristic.isLikelySpeedLimitCrop(crop.sign)) {
                continue;
            }

            if (isDuplicate(usedRects, crop.bounds)) {
                continue;
            }

            out.add(new Detection(crop.sign, crop.bounds));
            usedRects.add(crop.bounds);
        }
    }

    /**
     * Contours directement sur le masque binaire (plus fiable que Canny pour une zone rouge pleine).
     */
    private static List<MatOfPoint> findContoursOnMask(Mat mask) {
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        MatOfInt4 hierarchy = new MatOfInt4();
        Imgproc.findContours(
                mask.clone(),
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        );
        return contours;
    }

    private static void tryHoughCircles(Mat image, Mat hsv, List<Detection> out, List<Rect> usedRects) {
        tryHoughCircles(image, hsv, out, usedRects, MIN_SIGN_PX);
    }

    private static void tryHoughCircles(Mat image, Mat hsv, List<Detection> out, List<Rect> usedRects,
                                        int minSignPx) {
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(9, 9), 2);

        int minR = Math.max(8, Math.min(image.rows(), image.cols()) / 40);
        int maxR = Math.min(image.rows(), image.cols()) / 4;

        Mat circles = new Mat();
        Imgproc.HoughCircles(
                gray,
                circles,
                Imgproc.HOUGH_GRADIENT,
                1.2,
                minR * 2,
                100,
                30,
                minR,
                maxR
        );

        if (circles.cols() == 0) {
            return;
        }

        Mat redMask = redThreshold(hsv, 12, 165, 55);

        for (int i = 0; i < circles.cols(); i++) {
            double[] c = circles.get(0, i);
            int cx = (int) Math.round(c[0]);
            int cy = (int) Math.round(c[1]);
            int r = (int) Math.round(c[2]);

            if (!isRedCircle(redMask, cx, cy, r)) {
                continue;
            }

            int x = Math.max(cx - r, 0);
            int y = Math.max(cy - r, 0);
            int w = Math.min(2 * r, image.cols() - x);
            int h = Math.min(2 * r, image.rows() - y);

            if (w < minSignPx || h < minSignPx) {
                continue;
            }

            Rect safe = new Rect(x, y, w, h);
            if (isDuplicate(usedRects, safe)) {
                continue;
            }

            Mat sign = new Mat(image, safe);
            Mat copy = new Mat();
            sign.copyTo(copy);

            if (!minSignPxIsVideo(minSignPx) && !SignTypeHeuristic.isLikelySpeedLimitCrop(copy)) {
                continue;
            }

            out.add(new Detection(copy, safe));
            usedRects.add(safe);

            Imgproc.circle(image, new Point(cx, cy), r, new Scalar(255, 0, 0), 2);
            Imgproc.rectangle(image, new Point(x, y), new Point(x + w, y + h), new Scalar(0, 255, 0), 2);
        }
    }

    private static boolean minSignPxIsVideo(int minSignPx) {
        return minSignPx < MIN_SIGN_PX;
    }

    private static boolean isRedCircle(Mat redMask, int cx, int cy, int r) {
        if (r <= 0) {
            return false;
        }
        int samples = 0;
        int redHits = 0;

        for (int angle = 0; angle < 360; angle += 20) {
            double rad = Math.toRadians(angle);
            int x = cx + (int) (r * 0.85 * Math.cos(rad));
            int y = cy + (int) (r * 0.85 * Math.sin(rad));

            if (x < 0 || y < 0 || x >= redMask.cols() || y >= redMask.rows()) {
                continue;
            }
            samples++;
            if (redMask.get(y, x)[0] > 127) {
                redHits++;
            }
        }

        return samples > 0 && (double) redHits / samples >= 0.45;
    }

    private static boolean isDuplicate(List<Rect> used, Rect candidate) {
        for (Rect existing : used) {
            if (overlapRatio(existing, candidate) > 0.45) {
                return true;
            }
        }
        return false;
    }

    private static double overlapRatio(Rect a, Rect b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width);
        int y2 = Math.min(a.y + a.height, b.y + b.height);

        int interW = Math.max(0, x2 - x1);
        int interH = Math.max(0, y2 - y1);
        double inter = interW * interH;
        double union = a.width * a.height + b.width * b.height - inter;

        if (union <= 0) {
            return 0;
        }
        return inter / union;
    }

    public static Mat redThreshold(Mat hsv, int seuilRougeOrange, int seuilRougeViolet, int seuilSaturation) {
        Mat redMask1 = new Mat();
        Mat redMask2 = new Mat();
        Mat redMask = new Mat();

        Core.inRange(
                hsv,
                new Scalar(0, seuilSaturation, 40),
                new Scalar(seuilRougeOrange, 255, 255),
                redMask1
        );

        Core.inRange(
                hsv,
                new Scalar(seuilRougeViolet, seuilSaturation, 40),
                new Scalar(180, 255, 255),
                redMask2
        );

        Core.bitwise_or(redMask1, redMask2, redMask);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(redMask, redMask, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(redMask, redMask, Imgproc.MORPH_OPEN, kernel);

        return redMask;
    }

    private static class CropResult {
        final Mat sign;
        final Rect bounds;

        CropResult(Mat sign, Rect bounds) {
            this.sign = sign;
            this.bounds = bounds;
        }
    }

    private static CropResult extractCleanCircularSign(Mat originalImage, MatOfPoint contour) {
        Rect rect = Imgproc.boundingRect(contour);
        double contourArea = Imgproc.contourArea(contour);

        MatOfPoint2f contour2f = new MatOfPoint2f();
        contour2f.fromList(contour.toList());

        Point center = new Point();
        float[] radius = new float[1];
        Imgproc.minEnclosingCircle(contour2f, center, radius);

        if (radius[0] <= 0) {
            return null;
        }

        double circleArea = Math.PI * radius[0] * radius[0];
        if (circleArea <= 0) {
            return null;
        }

        double circularityRatio = contourArea / circleArea;
        if (circularityRatio < MIN_CIRCULARITY) {
            return null;
        }

        int pad = (int) Math.max(2, radius[0] * 0.08);
        int x = Math.max((int) center.x - (int) radius[0] - pad, 0);
        int y = Math.max((int) center.y - (int) radius[0] - pad, 0);
        int right = Math.min(x + 2 * ((int) radius[0] + pad), originalImage.cols());
        int bottom = Math.min(y + 2 * ((int) radius[0] + pad), originalImage.rows());

        if (right <= x || bottom <= y) {
            return null;
        }

        Rect safeRect = new Rect(x, y, right - x, bottom - y);
        Mat cleanCrop = new Mat(originalImage, safeRect);
        Mat sign = new Mat();
        cleanCrop.copyTo(sign);

        Imgproc.circle(originalImage, center, (int) radius[0], new Scalar(255, 0, 0), 2);
        Imgproc.rectangle(
                originalImage,
                new Point(safeRect.x, safeRect.y),
                new Point(safeRect.x + safeRect.width, safeRect.y + safeRect.height),
                new Scalar(0, 255, 0),
                2
        );

        return new CropResult(sign, safeRect);
    }

    public static void saveDetectedSigns(List<Mat> signs, String outputFolderPath) {
        File outputFolder = new File(outputFolderPath);
        outputFolder.mkdirs();

        for (int i = 0; i < signs.size(); i++) {
            File outputFile = new File(outputFolder, "detected_sign_" + (i + 1) + ".jpg");
            Imgcodecs.imwrite(outputFile.getAbsolutePath(), signs.get(i));
        }
    }
}
