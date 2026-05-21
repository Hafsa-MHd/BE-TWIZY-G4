import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.List;

/**
 * Photo externe avec CNN (ex. p5.jpg pour 110).
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

        List<Mat> signs = SignDetector.detectCircularRedSigns(scene);
        System.out.println("=== SceneRecognitionCnn ===");
        System.out.println("Image : " + imagePath);
        System.out.println("Panneaux : " + signs.size());

        File detectedDir = new File("detected_signs");
        detectedDir.mkdirs();

        for (int i = 0; i < signs.size(); i++) {
            File cropFile = new File(detectedDir, "sign_cnn_" + (i + 1) + ".jpg");
            Imgcodecs.imwrite(cropFile.getAbsolutePath(), signs.get(i));

            String type = SignTypeHeuristic.detectType(signs.get(i));
            if (!"SPEED".equals(type)) {
                type = "NON_SPEED";
            }

            String label = classifier.predict(cropFile, signs.get(i), type);
            System.out.println("Panneau " + (i + 1) + " : " + SignRecognitionPipeline.formatLabel(label));
            ImageUtils.show("CNN — " + label, signs.get(i));
        }
    }
}
