import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

/**
 * Analyse une vidéo sans fenêtre : affiche, toutes les 20 images,
 * la vitesse dominante (KNN ou SVM) pour déboguer video1.avi.
 *
 * Usage :
 *   VideoSpeedLog knn external_images/video1.avi
 *   VideoSpeedLog svm external_images/video1.avi
 *   VideoSpeedLog cnn external_images/video1.avi
 */
public class VideoSpeedLog {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String mode = args.length > 0 ? args[0].toLowerCase() : "knn";
        boolean useSvm = "svm".equals(mode);
        boolean useCnn = "cnn".equals(mode);
        String videoPath = args.length > 1 ? args[1]
                : (!useSvm && !useCnn && !"knn".equals(mode)
                ? args[0] : "external_images/video1.avi");

        String trainPath = "dataset_filtered/train";
        String modeLabel = useCnn ? "CNN" : (useSvm ? "SVM" : "KNN");
        System.out.println("Mode : " + modeLabel);
        System.out.println("Vidéo : " + videoPath);

        Object pipeline;
        if (useCnn) {
            pipeline = new SignRecognitionPipelineCnn(trainPath);
        } else if (useSvm) {
            pipeline = new SignRecognitionPipelineSvm(trainPath);
        } else {
            pipeline = new SignRecognitionPipeline(trainPath);
        }

        VideoSignTracker tracker = useCnn
                ? VideoSignTracker.forCnn() : VideoSignTracker.forClassic();
        VideoCapture cap = new VideoCapture(videoPath);
        if (!cap.isOpened()) {
            System.out.println("Impossible d'ouvrir : " + videoPath);
            return;
        }

        Mat frame = new Mat();
        int index = 0;
        int count50After240 = 0;

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }
            index++;

            int highwayStep = tracker.getHighwayStep();
            SignRecognitionPipeline.VideoFrameSummary summary;
            if (useCnn) {
                summary = ((SignRecognitionPipelineCnn) pipeline).processFrameForVideo(frame, highwayStep);
            } else if (useSvm) {
                summary = ((SignRecognitionPipelineSvm) pipeline).processFrameForVideo(frame, highwayStep);
            } else {
                summary = ((SignRecognitionPipeline) pipeline).processFrameForVideo(frame, highwayStep);
            }

            String display = summary.speedLabel != null
                    ? SignRecognitionPipeline.formatLabel(summary.speedLabel) : null;
            display = VideoSignTracker.filterSpeedForDisplay(display, highwayStep);
            if (display == null) {
                display = "-";
            }

            if (index > 260 && "50".equals(display)) {
                count50After240++;
                System.out.println("frame " + index + " : détection 50 (après zone 70)");
            }

            if (index % 20 == 0) {
                System.out.println("frame " + index + " : vitesse frame = " + display);
            }

            String announced = tracker.update(
                    summary.speedLabel != null
                            ? VideoSignTracker.filterSpeedForDisplay(
                                    SignRecognitionPipeline.formatLabel(summary.speedLabel), highwayStep)
                            : null);
            if (announced != null) {
                System.out.println(">>> annonce frame " + index + " : " + announced);
            }
        }

        cap.release();
        System.out.println("Total frames : " + index);
        System.out.println("Détections 50 après frame 260 : " + count50After240);
    }
}
