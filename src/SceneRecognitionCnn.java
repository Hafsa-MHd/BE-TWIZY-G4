import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.List;

/**
 * Photo externe avec CNN (ronds + triangles).
 */
public class SceneRecognitionCnn {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String trainPath = "dataset_filtered/train";
        String imagePath = args.length > 0 ? args[0] : "external_images/p10.jpg";

        SignClassifierCnn classifier = new SignClassifierCnn();
        if (!classifier.loadModels()) {
            System.out.println("Lancez d'abord : py -3 scripts/train_cnn.py");
            return;
        }
        classifier.trainRefineKnn(trainPath);

        Mat scene = Imgcodecs.imread(imagePath);
        if (scene.empty()) {
            System.out.println("Image introuvable : " + imagePath);
            return;
        }

        List<SignDetector.Detection> detections = SignDetector.detectDetailed(scene);
        System.out.println("=== SceneRecognitionCnn ===");
        System.out.println("Image : " + imagePath);
        System.out.println("Panneaux : " + detections.size());

        File detectedDir = new File("detected_signs");
        detectedDir.mkdirs();

        int index = 0;
        for (SignDetector.Detection detection : detections) {
            index++;
            Mat sign = detection.getCrop();
            File cropFile = new File(detectedDir, "sign_cnn_" + index + ".jpg");
            Imgcodecs.imwrite(cropFile.getAbsolutePath(), sign);

            String type = detection.isTriangular() ? "NON_SPEED" : SignTypeHeuristic.detectType(sign);
            if (!detection.isTriangular() && !"SPEED".equals(type)) {
                type = "NON_SPEED";
            }

            String label = classifier.predict(cropFile, sign, type);
            System.out.println("Panneau " + index + " ("
                    + (detection.isTriangular() ? "triangle" : "rond") + ") : "
                    + SignRecognitionPipeline.formatLabel(label));
            ImageUtils.show("CNN — " + label, sign);
        }
    }
}
