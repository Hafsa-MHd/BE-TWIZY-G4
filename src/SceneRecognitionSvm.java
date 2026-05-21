import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.List;

/**
 * Même scène que {@link SceneRecognition}, mais classificateur SVM.
 */
public class SceneRecognitionSvm {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String trainPath = "dataset_filtered/train";
        String imagePath = args.length > 0 ? args[0] : "external_images/tr2.jpg";

        SignClassifierSvm classifier = new SignClassifierSvm();
        classifier.prepare(trainPath);
        classifier.train(trainPath);

        Mat scene = Imgcodecs.imread(imagePath);
        if (scene.empty()) {
            System.out.println("Image introuvable : " + imagePath);
            return;
        }

        List<SignDetector.Detection> detections = SignDetector.detectDetailed(scene);
        System.out.println("=== SceneRecognitionSvm ===");
        System.out.println("Image : " + imagePath);
        System.out.println("Panneaux détectés : " + detections.size());

        File detectedDir = new File("detected_signs");
        detectedDir.mkdirs();

        int index = 0;
        for (SignDetector.Detection detection : detections) {
            index++;
            Mat sign = detection.getCrop();
            File cropFile = new File(detectedDir, "sign_svm_" + index + ".jpg");
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
            System.out.println("Classe SVM: " + label);
            System.out.println("Affichage : " + SignRecognitionPipeline.formatLabel(label));

            ImageUtils.show("SVM — Panneau " + index + " : " + label, sign);
        }

        File resultsDir = new File("results");
        resultsDir.mkdirs();
        Imgcodecs.imwrite(new File(resultsDir, "scene_annotated_svm.jpg").getAbsolutePath(), scene);

        System.out.println("\nPanneaux extraits : detected_signs/sign_svm_*.jpg");
        System.out.println("Scène annotée     : results/scene_annotated_svm.jpg");
    }
}
