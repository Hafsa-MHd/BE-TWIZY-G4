import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Même pipeline que {@link SignRecognitionPipeline}, mais avec {@link SignClassifierSvm}.
 */
public class SignRecognitionPipelineSvm {

    private final SignClassifierSvm classifier;

    public SignRecognitionPipelineSvm(String trainPath) {
        classifier = new SignClassifierSvm();
        classifier.prepare(trainPath);
        classifier.train(trainPath);
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

        int drawn = 0;
        for (SignDetector.Detection detection : detections) {
            if (!isRelevantDetection(detection, frame.cols(), frame.rows(), true)) {
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
            drawn++;
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
        Point origin = new Point(bounds.x, y);

        Imgproc.putText(
                frame,
                text,
                origin,
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.65,
                new Scalar(255, 200, 0),
                2,
                Imgproc.LINE_AA
        );
    }
}
