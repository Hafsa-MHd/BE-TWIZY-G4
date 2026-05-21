import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.List;

/**
 * Pipeline vidéo / image avec {@link SignClassifierCnn} (ONNX).
 */
public class SignRecognitionPipelineCnn {

    private final SignClassifierCnn classifier;

    public SignRecognitionPipelineCnn(String trainPath) {
        classifier = new SignClassifierCnn();
        if (!classifier.loadModels()) {
            throw new IllegalStateException(
                    "Modèles CNN introuvables. Lancez : py -3 scripts/train_cnn.py");
        }
        classifier.trainRefineKnn(trainPath);
    }

    public SignRecognitionPipeline.VideoFrameSummary processFrameForVideo(Mat frame) {
        return processFrameForVideo(frame, 0);
    }

    public SignRecognitionPipeline.VideoFrameSummary processFrameForVideo(Mat frame, int highwayStep) {
        SignRecognitionPipeline.VideoFrameSummary summary = new SignRecognitionPipeline.VideoFrameSummary();
        if (frame == null || frame.empty()) {
            return summary;
        }

        List<SignDetector.Detection> detections = SignDetector.detectDetailed(frame, true);
        String bestSpeed = null;
        int bestSpeedArea = 0;
        String bestOther = null;
        int bestOtherArea = 0;
        int frameArea = frame.cols() * frame.rows();

        for (SignDetector.Detection detection : detections) {
            if (!isRelevantDetection(detection, frame.cols(), frame.rows())) {
                continue;
            }

            String label = classifyDetection(detection, frameArea, true);
            summary.allLabels.add(label);

            Rect bounds = detection.getBounds();
            int area = bounds.width * bounds.height;

            if (label.startsWith("Speed_limit_")) {
                if (highwayStep < 2 && "Speed_limit_50_km_h".equals(label)) {
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

            drawLabel(frame, bounds, label);
        }

        summary.speedLabel = bestSpeed;
        summary.otherLabel = bestOther;
        return summary;
    }

    private String classifyDetection(SignDetector.Detection detection, int frameArea, boolean forVideo) {
        Mat crop = detection.getCrop();
        Rect bounds = detection.getBounds();
        double areaRatio = (bounds.width * bounds.height) / (double) Math.max(frameArea, 1);

        String type = SignTypeHeuristic.detectType(crop, false);
        if (SignTypeHeuristic.looksLikeSpeedLimitSign(crop)) {
            type = "SPEED";
        } else if (!"SPEED".equals(type)) {
            type = "NON_SPEED";
        }

        return classifier.predict(crop, type, forVideo, areaRatio);
    }

    private static boolean isRelevantDetection(SignDetector.Detection detection, int frameW, int frameH) {
        Rect r = detection.getBounds();
        int area = r.width * r.height;
        int frameArea = frameW * frameH;

        if (area < frameArea * 0.0007 || area > frameArea * 0.20) {
            return false;
        }
        if (r.width < 14 || r.height < 14) {
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

    private static void drawLabel(Mat frame, Rect bounds, String label) {
        String text = SignRecognitionPipeline.formatLabel(label);
        int y = Math.max(bounds.y - 8, 20);
        Imgproc.putText(
                frame,
                text,
                new Point(bounds.x, y),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.65,
                new Scalar(255, 0, 255),
                2,
                Imgproc.LINE_AA
        );
    }
}
