import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import java.awt.image.BufferedImage;

/**
 * Détection et classification de panneaux sur un flux vidéo (même pipeline KNN que les images).
 *
 * Vidéo attendue (ex. video1.avi) : panneaux 90 → 70 → 50 puis Dangerous_curve_to_the_right.
 * Lancement : {@code VideoRecognition external_images/video1.avi}
 */
public class VideoRecognition {

    private static final String DEFAULT_TRAIN = "dataset_filtered/train";
    private static final String DEFAULT_VIDEO = "external_images/video1.avi";
    private static final int FRAME_DELAY_MS = 30;

    public static void main(String[] args) {
        System.out.println("Bibliothèque OpenCV : " + Core.NATIVE_LIBRARY_NAME);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String videoPath = args.length > 0 ? args[0] : DEFAULT_VIDEO;
        String trainPath = args.length > 1 ? args[1] : DEFAULT_TRAIN;

        System.out.println("=== Entraînement (une seule fois) ===");
        SignRecognitionPipeline pipeline = new SignRecognitionPipeline(trainPath);

        VideoSignTracker speedTracker = VideoSignTracker.forClassic();
        VideoSignTracker otherTracker = VideoSignTracker.forClassic();

        JFrame frame = new JFrame("Détection de panneaux — " + videoPath);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JLabel videoPanel = new JLabel();
        frame.setContentPane(videoPanel);
        frame.setSize(960, 600);
        frame.setVisible(true);

        VideoCapture camera = new VideoCapture(videoPath);
        if (!camera.isOpened()) {
            System.out.println("Erreur : impossible d'ouvrir " + videoPath);
            return;
        }

        Mat image = new Mat();
        int frameIndex = 0;

        System.out.println("\n=== Lecture vidéo ===");
        System.out.println("Fichier : " + videoPath);
        System.out.println("(Annonce quand un libellé domine ~7/15 images — 90, 70, 50…)");
        System.out.println("Debug : VideoSpeedLog knn " + videoPath + "\n");

        while (camera.read(image)) {
            if (image.empty()) {
                break;
            }

            frameIndex++;
            int highwayStep = speedTracker.getHighwayStep();
            SignRecognitionPipeline.VideoFrameSummary summary =
                    pipeline.processFrameForVideo(image, highwayStep);

            String speedDisplay = summary.speedLabel != null
                    ? SignRecognitionPipeline.formatLabel(summary.speedLabel) : null;
            speedDisplay = VideoSignTracker.filterSpeedForDisplay(speedDisplay, highwayStep);
            String otherDisplay = summary.otherLabel != null
                    ? SignRecognitionPipeline.formatLabel(summary.otherLabel) : null;

            String announcedSpeed = speedTracker.update(speedDisplay);
            if (announcedSpeed != null) {
                System.out.println(">>> Panneau vitesse : " + announcedSpeed);
            }

            String announcedOther = otherTracker.update(otherDisplay);
            if (announcedOther != null) {
                System.out.println(">>> Panneau signalisation : " + announcedOther);
            }

            BufferedImage display = ImageUtils.matToBufferedImage(image);
            if (display != null) {
                videoPanel.setIcon(new ImageIcon(display));
                videoPanel.repaint();
            }

            try {
                Thread.sleep(FRAME_DELAY_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        camera.release();
        System.out.println("\nFin de la vidéo.");
        System.out.println("Images traitées : " + frameIndex);
    }
}
