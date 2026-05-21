import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.videoio.VideoCapture;

import java.util.List;

/** Dump détections brutes par frame (sans filtre tracker). */
public class VideoFrameDump {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        String video = args.length > 0 ? args[0] : "external_images/video1.avi";

        VideoCapture cap = new VideoCapture(video);
        Mat frame = new Mat();
        int i = 0;

        while (cap.read(frame)) {
            i++;
            if (i < 250 || i > 395 || i % 10 != 0) {
                continue;
            }
            List<SignDetector.Detection> all = SignDetector.detectDetailed(frame, true);
            int kept = 0;
            StringBuilder sb = new StringBuilder("frame " + i + " : " + all.size() + " brut");
            for (SignDetector.Detection d : all) {
                boolean rel = SignRecognitionPipelineSvmIsRelevant.isRelevant(
                        d, frame.cols(), frame.rows());
                if (rel) {
                    kept++;
                }
            }
            sb.append(", ").append(kept).append(" filtré");
            System.out.println(sb);
        }
        cap.release();
    }

    /** Copie des critères de SignRecognitionPipelineSvm. */
    static class SignRecognitionPipelineSvmIsRelevant {
        static boolean isRelevant(SignDetector.Detection detection, int frameW, int frameH) {
            Rect r = detection.getBounds();
            int area = r.width * r.height;
            int frameArea = frameW * frameH;
            if (area < frameArea * 0.0015 || area > frameArea * 0.20) {
                return false;
            }
            if (r.width < 20 || r.height < 20) {
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
    }
}
