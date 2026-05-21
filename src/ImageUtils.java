import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * Utilitaires OpenCV partagés (affichage, HSV, contours).
 */
public class ImageUtils {

    public static Mat bgrToHsv(Mat bgr) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV);
        return hsv;
    }

    public static List<MatOfPoint> findContours(Mat binaryOrMask) {
        Mat edges = new Mat();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        MatOfInt4 hierarchy = new MatOfInt4();

        int thresh = 100;
        Imgproc.Canny(binaryOrMask, edges, thresh, thresh * 2);
        Imgproc.findContours(
                edges,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        return contours;
    }

    public static void show(String title, Mat image) {
        if (image == null || image.empty()) {
            System.out.println("Impossible d'afficher l'image : image vide.");
            return;
        }

        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", image, buffer);

        try {
            InputStream in = new ByteArrayInputStream(buffer.toArray());
            BufferedImage buf = ImageIO.read(in);

            JFrame frame = new JFrame(title);
            frame.getContentPane().add(new JLabel(new ImageIcon(buf)));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static boolean isImageFile(java.io.File file) {
        String name = file.getName().toLowerCase();
        return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png");
    }

    /** Conversion Mat OpenCV → BufferedImage pour affichage Swing (flux vidéo). */
    public static BufferedImage matToBufferedImage(Mat image) {
        if (image == null || image.empty()) {
            return null;
        }
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".jpg", image, buffer);
        try {
            InputStream in = new ByteArrayInputStream(buffer.toArray());
            return ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
