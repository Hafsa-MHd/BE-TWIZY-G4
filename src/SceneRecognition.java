import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.List;

/**
 * Photo : panneaux ronds rouges + panneaux triangulaires jaunes, puis classification KNN.
 */
public class SceneRecognition {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String trainPath = "dataset_filtered/train";
        String imagePath = args.length > 0 ? args[0] : "external_images/tr1.jpg";

        SignClassifier classifier = new SignClassifier();
        classifier.prepare(trainPath);
        classifier.train(trainPath);

        Mat scene = Imgcodecs.imread(imagePath);
        if (scene.empty()) {
            System.out.println("Image introuvable : " + imagePath);
            return;
        }

        List<SignDetector.Detection> detections = SignDetector.detectDetailed(scene);
        System.out.println("Image : " + imagePath);
        System.out.println("Panneaux détectés : " + detections.size());

        File detectedDir = new File("detected_signs");
        detectedDir.mkdirs();

        int index = 0;
        for (SignDetector.Detection detection : detections) {
            index++;
            Mat sign = detection.getCrop();
            File cropFile = new File(detectedDir, "sign_" + index + ".jpg");
            Imgcodecs.imwrite(cropFile.getAbsolutePath(), sign);

            String type = detection.isTriangular() ? "NON_SPEED" : SignTypeHeuristic.detectType(sign);
            if (!detection.isTriangular() && !"SPEED".equals(type)) {
                type = "NON_SPEED";
            }
            if (!detection.isTriangular() && SignTypeHeuristic.looksLikeSpeedLimitSign(sign)) {
                type = "SPEED";
            }

            String label = classifier.predict(cropFile, sign, type);

            System.out.println("--------------------------------");
            System.out.println("Panneau " + index + " : " + cropFile.getName());
            System.out.println("Forme     : " + (detection.isTriangular() ? "triangle" : "rond"));
            System.out.println("Type      : " + type);
            System.out.println("Classe    : " + label);

            ImageUtils.show("Panneau " + index + " : " + label, sign);
        }

        File resultsDir = new File("results");
        resultsDir.mkdirs();
        Imgcodecs.imwrite(new File(resultsDir, "scene_annotated.jpg").getAbsolutePath(), scene);

        System.out.println("\nPanneaux extraits : detected_signs/");
        System.out.println("Scène annotée     : results/scene_annotated.jpg");
    }
}
