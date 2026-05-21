import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Classificateur CNN (modèles ONNX entraînés par scripts/train_cnn.py).
 * KNN + SVM inchangés — branche parallèle pour comparaison.
 */
public class SignClassifierCnn {

    private static final int K_REFINE = 5;
    private static final int MAX_IMAGES_PER_CLASS = 200;
    private static final String MODELS_DIR = "models";
    private static final String SPEED_ONNX = MODELS_DIR + "/speed_cnn.onnx";
    private static final String OTHER_ONNX = MODELS_DIR + "/other_cnn.onnx";
    private static final String SPEED_LABELS = MODELS_DIR + "/speed_cnn_labels.txt";
    private static final String OTHER_LABELS = MODELS_DIR + "/other_cnn_labels.txt";

    private final ArrayList<String> speedLabels = new ArrayList<String>();
    private final ArrayList<String> otherLabels = new ArrayList<String>();

    private Net speedNet;
    private Net otherNet;
    private KNearest speedKnnRefine;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        SignClassifierCnn classifier = new SignClassifierCnn();
        if (!classifier.loadModels()) {
            System.out.println("Modèles CNN absents. Entraînez d'abord :");
            System.out.println("  py -3 scripts/train_cnn.py");
            return;
        }

        classifier.trainRefineKnn("dataset_filtered/train");

        System.out.println("\n=== Évaluation CNN sur le test ===");
        classifier.evaluate("dataset_filtered/test");
        System.out.println("\nRapports : results/*_cnn.csv");
    }

    public boolean loadModels() {
        File speedFile = new File(SPEED_ONNX);
        File otherFile = new File(OTHER_ONNX);
        if (!speedFile.isFile() || !otherFile.isFile()) {
            return false;
        }

        speedNet = Dnn.readNetFromONNX(speedFile.getAbsolutePath());
        otherNet = Dnn.readNetFromONNX(otherFile.getAbsolutePath());
        speedNet.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        speedNet.setPreferableTarget(Dnn.DNN_TARGET_CPU);
        otherNet.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        otherNet.setPreferableTarget(Dnn.DNN_TARGET_CPU);

        loadLabelList(SPEED_LABELS, speedLabels);
        loadLabelList(OTHER_LABELS, otherLabels);

        System.out.println("CNN vitesse chargé : " + speedLabels.size() + " classes");
        System.out.println("CNN autres chargé  : " + otherLabels.size() + " classes");
        return speedLabels.size() > 0 && otherLabels.size() > 0;
    }

    /** KNN HOG auxiliaire pour SpeedLimitRefinement (comme SVM). */
    public void trainRefineKnn(String trainPath) {
        HashMap<String, Integer> speedLabelToId = new HashMap<String, Integer>();
        for (int i = 0; i < speedLabels.size(); i++) {
            speedLabelToId.put(speedLabels.get(i), i);
        }
        ArrayList<Sample> speedSamples = loadSamples(trainPath, true, speedLabelToId, MAX_IMAGES_PER_CLASS);
        speedKnnRefine = trainKnnRefine(speedSamples);
        if (speedKnnRefine != null) {
            System.out.println("KNN auxiliaire vitesse (raffinement) entraîné.");
        }
    }

    public String predict(Mat signCrop, String signType) {
        return predict(signCrop, signType, false, 0.0);
    }

    public String predict(Mat signCrop, String signType, boolean forVideo, double signAreaRatio) {
        if (signCrop == null || signCrop.empty()) {
            return "IMAGE_NON_TRAITEE";
        }
        if ("SPEED".equals(signType)) {
            String label = predictFromMat(speedNet, signCrop, speedLabels);
            return SpeedLimitRefinement.refine(signCrop, label, speedKnnRefine, null, speedLabels,
                    forVideo, signAreaRatio);
        }
        String label = predictFromMat(otherNet, signCrop, otherLabels);
        return SignTypeHeuristic.refineOtherLabel(signCrop, label, forVideo);
    }

    public String predict(File imageFile, Mat signCrop, String signType) {
        if ("SPEED".equals(signType)) {
            Mat img = org.opencv.imgcodecs.Imgcodecs.imread(imageFile.getAbsolutePath());
            String label = predictFromMat(speedNet, img, speedLabels);
            if (signCrop != null && !signCrop.empty()) {
                label = SpeedLimitRefinement.refine(signCrop, label, speedKnnRefine, imageFile, speedLabels);
            }
            return label;
        }
        Mat img = org.opencv.imgcodecs.Imgcodecs.imread(imageFile.getAbsolutePath());
        String label = predictFromMat(otherNet, img, otherLabels);
        if (signCrop != null && !signCrop.empty()) {
            label = SignTypeHeuristic.refineOtherLabel(signCrop, label);
        }
        return label;
    }

    public void evaluate(String testPath) {
        File resultsDir = new File("results");
        resultsDir.mkdirs();
        evaluateSplit(testPath, true, speedNet, speedLabels,
                new File(resultsDir, "speed_limit_test_cnn.csv"), "Limitation de vitesse (CNN)");
        evaluateSplit(testPath, false, otherNet, otherLabels,
                new File(resultsDir, "other_signs_test_cnn.csv"), "Autres panneaux (CNN)");
    }

    private void evaluateSplit(String testPath, boolean speedOnly, Net net, ArrayList<String> labels,
                               File reportFile, String title) {
        HashMap<String, Integer> labelToId = new HashMap<String, Integer>();
        for (int i = 0; i < labels.size(); i++) {
            labelToId.put(labels.get(i), i);
        }

        ArrayList<Sample> samples = loadSamples(testPath, speedOnly, labelToId, Integer.MAX_VALUE);
        int total = 0;
        int correct = 0;
        Map<String, Integer> totalByClass = new TreeMap<String, Integer>();
        Map<String, Integer> correctByClass = new TreeMap<String, Integer>();

        try (PrintWriter writer = new PrintWriter(reportFile)) {
            writer.println("filename,true_label,predicted_label,result");
            for (Sample sample : samples) {
                Mat img = org.opencv.imgcodecs.Imgcodecs.imread(sample.file.getAbsolutePath());
                String predicted = predictFromMat(net, img, labels);
                String trueLabel = labels.get(sample.labelId);
                boolean ok = predicted.equals(trueLabel);
                total++;
                bump(totalByClass, trueLabel);
                if (ok) {
                    correct++;
                    bump(correctByClass, trueLabel);
                }
                writer.println(sample.file.getName() + "," + trueLabel + "," + predicted + ","
                        + (ok ? "OK" : "ERREUR"));
            }
        } catch (Exception e) {
            System.out.println("Erreur rapport : " + e.getMessage());
            return;
        }

        double accuracy = total > 0 ? (100.0 * correct) / total : 0.0;
        System.out.println("\n--- " + title + " ---");
        System.out.println("Images testées : " + total);
        System.out.printf("Précision      : %.1f %%\n", accuracy);
        System.out.println("Rapport        : " + reportFile.getAbsolutePath());
    }

    private String predictFromMat(Net net, Mat signCrop, ArrayList<String> labels) {
        if (net == null || signCrop == null || signCrop.empty()) {
            return "IMAGE_NON_TRAITEE";
        }
        Mat blob = CnnImageFeatures.blobFromSign(signCrop);
        if (blob.empty()) {
            return "IMAGE_NON_TRAITEE";
        }
        net.setInput(blob);
        Mat out = net.forward();
        int id = argmax(out);
        if (id >= 0 && id < labels.size()) {
            return labels.get(id);
        }
        return "INCONNU";
    }

    /** Lit tous les scores softmax (forme OpenCV variable : 1×N, 1×1×N…). */
    private static int argmax(Mat prob) {
        if (prob.empty()) {
            return 0;
        }
        int n = (int) prob.total();
        float[] scores = new float[n];
        prob.get(0, 0, scores);
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (scores[i] > scores[best]) {
                best = i;
            }
        }
        if (best >= n) {
            best = 0;
        }
        return best;
    }

    private static void loadLabelList(String path, ArrayList<String> out) {
        out.clear();
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    out.add(line);
                }
            }
        } catch (Exception e) {
            System.out.println("Erreur lecture " + path + " : " + e.getMessage());
        }
    }

    private KNearest trainKnnRefine(ArrayList<Sample> samples) {
        if (samples.isEmpty()) {
            return null;
        }
        Mat first = SpeedLimitFeatures.toHog(samples.get(0).file);
        if (first.empty()) {
            return null;
        }
        int featureLength = first.cols();
        Mat trainData = new Mat(samples.size(), featureLength, org.opencv.core.CvType.CV_32F);
        Mat trainLabels = new Mat(samples.size(), 1, org.opencv.core.CvType.CV_32F);
        for (int i = 0; i < samples.size(); i++) {
            Mat feature = SpeedLimitFeatures.toHog(samples.get(i).file);
            if (!feature.empty()) {
                feature.copyTo(trainData.row(i));
                trainLabels.put(i, 0, samples.get(i).labelId);
            }
        }
        KNearest knn = KNearest.create();
        knn.setDefaultK(K_REFINE);
        knn.setIsClassifier(true);
        if (!knn.train(trainData, Ml.ROW_SAMPLE, trainLabels)) {
            return null;
        }
        return knn;
    }

    private static ArrayList<Sample> loadSamples(String splitPath, boolean speedOnly,
                                                 HashMap<String, Integer> labelToId,
                                                 int maxPerClass) {
        ArrayList<Sample> samples = new ArrayList<Sample>();
        File splitFolder = new File(splitPath);
        File[] classFolders = splitFolder.listFiles();
        if (classFolders == null) {
            return samples;
        }
        for (File classFolder : classFolders) {
            if (!classFolder.isDirectory()) {
                continue;
            }
            String label = classFolder.getName();
            boolean isSpeed = label.startsWith("Speed_limit_");
            if (speedOnly != isSpeed || !labelToId.containsKey(label)) {
                continue;
            }
            int labelId = labelToId.get(label);
            File[] images = classFolder.listFiles();
            if (images == null) {
                continue;
            }
            int count = 0;
            for (File image : images) {
                if (ImageUtils.isImageFile(image)) {
                    samples.add(new Sample(image, labelId));
                    count++;
                    if (count >= maxPerClass) {
                        break;
                    }
                }
            }
        }
        return samples;
    }

    private static void bump(Map<String, Integer> map, String key) {
        map.put(key, map.getOrDefault(key, 0) + 1);
    }

    static class Sample {
        final File file;
        final int labelId;

        Sample(File file, int labelId) {
            this.file = file;
            this.labelId = labelId;
        }
    }
}
