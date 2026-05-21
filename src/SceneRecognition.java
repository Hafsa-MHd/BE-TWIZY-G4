import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.List;

/**
 * Pipeline complet : détecter les panneaux rouges circulaires dans une photo,
 * puis classer chaque panneau (vitesse vs autres).
 *
 * Modifier imagePath ci-dessous, puis lancer ce programme.
 */
public class SceneRecognition {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String trainPath = "dataset_filtered/train";
        String imagePath = "external_images/p7.JPG";

        SignClassifier classifier = new SignClassifier();
        classifier.prepare(trainPath);
        classifier.train(trainPath);

        Mat scene = Imgcodecs.imread(imagePath);
        if (scene.empty()) {
            System.out.println("Image introuvable : " + imagePath);
            return;
        }

        List<Mat> signs = SignDetector.detectCircularRedSigns(scene);
        System.out.println("Image : " + imagePath);
        System.out.println("Panneaux détectés : " + signs.size());

        File detectedDir = new File("detected_signs");
        detectedDir.mkdirs();

        for (int i = 0; i < signs.size(); i++) {
            File cropFile = new File(detectedDir, "sign_" + (i + 1) + ".jpg");
            Imgcodecs.imwrite(cropFile.getAbsolutePath(), signs.get(i));

            String type = SignTypeHeuristic.detectType(signs.get(i));
            if (!"SPEED".equals(type)) {
                type = "NON_SPEED";
            }

            String label = classifier.predict(cropFile, signs.get(i), type);

            System.out.println("--------------------------------");
            System.out.println("Panneau " + (i + 1) + " : " + cropFile.getName());
            System.out.println("Type      : " + type);
            System.out.println("Classe    : " + label);

            ImageUtils.show("Panneau " + (i + 1) + " : " + label, signs.get(i));
        }

        File resultsDir = new File("results");
        resultsDir.mkdirs();
        Imgcodecs.imwrite(new File(resultsDir, "scene_annotated.jpg").getAbsolutePath(), scene);

        System.out.println("\nPanneaux extraits : detected_signs/");
        System.out.println("Scène annotée     : results/scene_annotated.jpg");
    }
}
