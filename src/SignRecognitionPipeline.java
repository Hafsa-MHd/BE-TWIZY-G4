import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Même pipeline que {@link SceneRecognition}, réutilisable image par image ou sur une vidéo.
 */
public class SignRecognitionPipeline {

    private final SignClassifier classifier;

    public SignRecognitionPipeline(String trainPath) {
        classifier = new SignClassifier();
        classifier.prepare(trainPath);
        classifier.train(trainPath);
    }

    public static class FrameResult {
        public final List<SignDetector.Detection> detections = new ArrayList<SignDetector.Detection>();
        public final List<String> labels = new ArrayList<String>();
    }

    /** Résumé vidéo : meilleur panneau vitesse + meilleur autre panneau sur l'image. */
    public static class VideoFrameSummary {
        public String speedLabel;
        public String otherLabel;
        public final List<String> allLabels = new ArrayList<String>();
    }

    public FrameResult processFrame(Mat frame) {
        return processFrame(frame, false);
    }

    public FrameResult processFrame(Mat frame, boolean logToConsole) {
        FrameResult result = new FrameResult();
        if (frame == null || frame.empty()) {
            return result;
        }

        List<SignDetector.Detection> detections = SignDetector.detectDetailed(frame);
        int index = 0;
        for (SignDetector.Detection detection : detections) {
            if (!isRelevantDetection(detection, frame.cols(), frame.rows())) {
                continue;
            }

            int frameArea = frame.cols() * frame.rows();
            String label = classifyDetection(detection, true, frameArea, false);
            result.detections.add(detection);
            result.labels.add(label);

            drawLabel(frame, detection.getBounds(), label, index + 1);
            index++;

            if (logToConsole) {
                System.out.println("  Panneau " + index + " : " + formatLabel(label));
            }
        }

        return result;
    }

    /**
     * Mode vidéo : moins de bruit console, filtre les faux panneaux, retourne le meilleur vitesse / autre.
     */
    public VideoFrameSummary processFrameForVideo(Mat frame) {
        return processFrameForVideo(frame, 0);
    }

    /**
     * @param highwayStep étape séquence 90→70→50 ; si &lt; 2, n'affiche pas un faux 50 avant le 70.
     */
    public VideoFrameSummary processFrameForVideo(Mat frame, int highwayStep) {
        VideoFrameSummary summary = new VideoFrameSummary();
        if (frame == null || frame.empty()) {
            return summary;
        }

        List<SignDetector.Detection> detections = SignDetector.detectDetailed(frame, true);
        String bestSpeed = null;
        int bestSpeedArea = 0;
        String bestOther = null;
        int bestOtherArea = 0;
        int frameArea = frame.cols() * frame.rows();

        int drawn = 0;
        for (SignDetector.Detection detection : detections) {
            if (!isRelevantDetection(detection, frame.cols(), frame.rows(), true)) {
                continue;
            }

            String label = classifyDetection(detection, false, frameArea, true);
            summary.allLabels.add(label);

            Rect bounds = detection.getBounds();
            int area = bounds.width * bounds.height;

            if (label.startsWith("Speed_limit_")) {
                if (shouldIgnoreSpeedLabel(label, highwayStep)) {
                    continue;
                }
                if (area > bestSpeedArea) {
                    bestSpeedArea = area;
                    bestSpeed = label;
                }
            } else if (isImportantOtherSign(label)) {
                if (area > bestOtherArea) {
                    bestOtherArea = area;
                    bestOther = label;
                }
            }

            drawLabel(frame, bounds, label, drawn + 1);
            drawn++;
        }

        summary.speedLabel = bestSpeed;
        summary.otherLabel = bestOther;
        return summary;
    }

    private String classifyDetection(SignDetector.Detection detection, boolean logHeuristic,
                                     int frameArea, boolean forVideo) {
        Mat crop = detection.getCrop();
        Rect bounds = detection.getBounds();
        double areaRatio = (bounds.width * bounds.height) / (double) Math.max(frameArea, 1);

        String type = SignTypeHeuristic.detectType(crop, logHeuristic);
        if (SignTypeHeuristic.looksLikeSpeedLimitSign(crop)) {
            type = "SPEED";
        } else if (!"SPEED".equals(type)) {
            type = "NON_SPEED";
        }

        return classifier.predict(crop, type, forVideo, areaRatio);
    }

    private static boolean isRelevantDetection(SignDetector.Detection detection, int frameW, int frameH) {
        return isRelevantDetection(detection, frameW, frameH, false);
    }

    private static boolean isRelevantDetection(SignDetector.Detection detection, int frameW, int frameH,
                                              boolean forVideo) {
        Rect r = detection.getBounds();
        int area = r.width * r.height;
        int frameArea = frameW * frameH;
        double minArea = forVideo ? 0.0007 : 0.0015;
        int minSide = forVideo ? 14 : 20;

        if (area < frameArea * minArea) {
            return false;
        }
        if (area > frameArea * 0.20) {
            return false;
        }
        if (r.width < minSide || r.height < minSide) {
            return false;
        }

        Mat crop = detection.getCrop();
        SignTypeHeuristic.ColorProfile p = SignTypeHeuristic.analyzeCenter(crop);
        if (p == null) {
            return false;
        }

        if (SignTypeHeuristic.looksLikeSpeedLimitSign(crop)) {
            return true;
        }

        if (p.redRatio > 0.04 && p.blackRatio > 0.04) {
            return true;
        }

        return p.blackRatio > 0.12 || p.redRatio > 0.06;
    }

    private static boolean shouldIgnoreSpeedLabel(String label, int highwayStep) {
        return highwayStep < 2 && "Speed_limit_50_km_h".equals(label);
    }

    private static boolean isImportantOtherSign(String label) {
        if (label == null) {
            return false;
        }
        return label.startsWith("Dangerous_curve")
                || label.startsWith("Turn_")
                || "Road_work".equals(label)
                || "General_caution".equals(label)
                || "Bumpy_road".equals(label);
    }

    private static void drawLabel(Mat frame, Rect bounds, String label, int number) {
        String text = formatLabel(label);
        int y = Math.max(bounds.y - 8, 20);
        Point origin = new Point(bounds.x, y);

        Imgproc.putText(
                frame,
                text,
                origin,
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.65,
                new Scalar(0, 255, 0),
                2,
                Imgproc.LINE_AA
        );
    }

    static String formatLabel(String label) {
        if (label == null) {
            return "?";
        }
        if (label.startsWith("Speed_limit_")) {
            return label.replace("Speed_limit_", "").replace("_km_h", "");
        }
        if (label.startsWith("Dangerous_curve_to_the_")) {
            return label.replace("Dangerous_curve_to_the_", "Curve ");
        }
        return label;
    }
}
