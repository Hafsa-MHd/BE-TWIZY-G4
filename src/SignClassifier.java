import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Entraîne et évalue deux modèles KNN :
 * - vitesse (HOG sur chiffres) pour les classes Speed_limit_*
 * - autres panneaux (HOG sur panneau entier)
 *
 * Lancer en tant que programme Java pour obtenir la précision sur dataset_filtered/test.
 */
public class SignClassifier {

    /** K=1 : classes de vitesse très proches (30, 50, 70…) */
    private static final int K_SPEED = 1;
    /** K=3 : plus de panneaux, légèrement plus de robustesse au bruit */
    private static final int K_OTHER = 3;
    private static final int MAX_IMAGES_PER_CLASS = 200;

    private final ArrayList<String> speedLabels = new ArrayList<String>();
    private final HashMap<String, Integer> speedLabelToId = new HashMap<String, Integer>();

    private final ArrayList<String> otherLabels = new ArrayList<String>();
    private final HashMap<String, Integer> otherLabelToId = new HashMap<String, Integer>();

    private KNearest speedModel;
    private KNearest otherModel;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String trainPath = "dataset_filtered/train";
        String testPath = "dataset_filtered/test";

        SignClassifier classifier = new SignClassifier();
        classifier.prepare(trainPath);

        System.out.println("=== Entraînement ===");
        classifier.train(trainPath);

        System.out.println("\n=== Évaluation sur le test ===");
        classifier.evaluate(testPath);

        System.out.println("\nTerminé. Rapports dans : results/");
    }

    public void prepare(String trainPath) {
        loadLabels(trainPath, true, speedLabels, speedLabelToId);
        loadLabels(trainPath, false, otherLabels, otherLabelToId);

        System.out.println("Classes vitesse : " + speedLabels.size());
        System.out.println("Classes autres  : " + otherLabels.size());
    }

    public void train(String trainPath) {
        ArrayList<Sample> speedSamples = loadSamples(trainPath, true, speedLabelToId, MAX_IMAGES_PER_CLASS);
        ArrayList<Sample> otherSamples = loadSamples(trainPath, false, otherLabelToId, MAX_IMAGES_PER_CLASS);

        System.out.println("Images vitesse (train) : " + speedSamples.size());
        System.out.println("Images autres (train)  : " + otherSamples.size());

        speedModel = trainModel(speedSamples, true, K_SPEED);
        otherModel = trainModel(otherSamples, false, K_OTHER);

        if (speedModel == null || otherModel == null) {
            throw new IllegalStateException("Échec d'entraînement — vérifiez dataset_filtered/train.");
        }
    }

    public String predict(File imageFile, String signType) {
        return predict(imageFile, null, signType);
    }

    /** Classification à partir d'un crop Mat (vidéo, sans fichier temporaire). */
    public String predict(Mat signCrop, String signType) {
        return predict(signCrop, signType, false);
    }

    /** @param forVideo true = pas de correction OCR agressive (flux vidéo) */
    public String predict(Mat signCrop, String signType, boolean forVideo) {
        return predict(signCrop, signType, forVideo, 0.0);
    }

    public String predict(Mat signCrop, String signType, boolean forVideo, double signAreaRatio) {
        if (signCrop == null || signCrop.empty()) {
            return "IMAGE_NON_TRAITEE";
        }
        if ("SPEED".equals(signType)) {
            String label = predictWithModelFromMat(speedModel, signCrop, true, speedLabels, K_SPEED);
            return SpeedLimitRefinement.refine(signCrop, label, speedModel, null, speedLabels,
                    forVideo, signAreaRatio);
        }
        String label = predictWithModelFromMat(otherModel, signCrop, false, otherLabels, K_OTHER);
        return SignTypeHeuristic.refineOtherLabel(signCrop, label, forVideo);
    }

    /**
     * @param signCrop image du panneau (pour raffinement heuristique), peut être null
     */
    public String predict(File imageFile, Mat signCrop, String signType) {
        if ("SPEED".equals(signType)) {
            String label = predictWithModel(speedModel, imageFile, true, speedLabels, K_SPEED);
            if (signCrop != null && !signCrop.empty()) {
                label = SpeedLimitRefinement.refine(signCrop, label, speedModel, imageFile, speedLabels);
            }
            return label;
        }

        String label = predictWithModel(otherModel, imageFile, false, otherLabels, K_OTHER);

        if (signCrop != null && !signCrop.empty()) {
            label = SignTypeHeuristic.refineOtherLabel(signCrop, label);
        }

        return label;
    }

    public void evaluate(String testPath) {
        File resultsDir = new File("results");
        resultsDir.mkdirs();

        evaluateSplit(testPath, true, speedModel, speedLabels,
                new File(resultsDir, "speed_limit_test.csv"), "Limitation de vitesse");

        evaluateSplit(testPath, false, otherModel, otherLabels,
                new File(resultsDir, "other_signs_test.csv"), "Autres panneaux");
    }

    private void evaluateSplit(
            String testPath,
            boolean speedOnly,
            KNearest model,
            ArrayList<String> labels,
            File reportFile,
            String title
    ) {
        ArrayList<Sample> samples = loadSamples(testPath, speedOnly,
                speedOnly ? speedLabelToId : otherLabelToId, Integer.MAX_VALUE);

        int total = 0;
        int correct = 0;
        Map<String, Integer> totalByClass = new TreeMap<String, Integer>();
        Map<String, Integer> correctByClass = new TreeMap<String, Integer>();

        try (PrintWriter writer = new PrintWriter(reportFile)) {
            writer.println("filename,true_label,predicted_label,result");

            for (Sample sample : samples) {
                int k = speedOnly ? K_SPEED : K_OTHER;
                String predicted = predictWithModel(model, sample.file, speedOnly, labels, k);
                String trueLabel = labels.get(sample.labelId);
                boolean ok = predicted.equals(trueLabel);

                total++;
                bump(totalByClass, trueLabel);
                if (ok) {
                    correct++;
                    bump(correctByClass, trueLabel);
                }

                writer.println(sample.file.getName() + ","
                        + trueLabel + "," + predicted + "," + (ok ? "OK" : "ERREUR"));
            }
        } catch (Exception e) {
            System.out.println("Erreur rapport : " + reportFile.getName());
            e.printStackTrace();
            return;
        }

        double accuracy = total > 0 ? (100.0 * correct) / total : 0.0;

        System.out.println("\n--- " + title + " ---");
        System.out.println("Images testées : " + total);
        System.out.println("Précision      : " + String.format("%.1f", accuracy) + " %");
        System.out.println("Rapport        : " + reportFile.getAbsolutePath());

        for (String label : totalByClass.keySet()) {
            int t = totalByClass.get(label);
            int c = correctByClass.containsKey(label) ? correctByClass.get(label) : 0;
            double acc = t > 0 ? (100.0 * c) / t : 0.0;
            System.out.println("  " + label + " : " + c + "/" + t + " = " + String.format("%.1f", acc) + " %");
        }
    }

    private KNearest trainModel(ArrayList<Sample> samples, boolean speedModel, int k) {
        if (samples.isEmpty()) {
            return null;
        }

        Mat first = extractFeature(samples.get(0).file, speedModel);
        if (first.empty()) {
            return null;
        }

        int featureLength = first.cols();
        Mat trainData = new Mat(samples.size(), featureLength, CvType.CV_32F);
        Mat trainLabels = new Mat(samples.size(), 1, CvType.CV_32F);
        first.release();

        for (int i = 0; i < samples.size(); i++) {
            Mat feature = extractFeature(samples.get(i).file, speedModel);
            if (!feature.empty()) {
                feature.copyTo(trainData.row(i));
                trainLabels.put(i, 0, samples.get(i).labelId);
            }
            feature.release();
        }

        KNearest knn = KNearest.create();
        knn.setDefaultK(k);
        knn.setIsClassifier(true);

        if (!knn.train(trainData, Ml.ROW_SAMPLE, trainLabels)) {
            return null;
        }

        System.out.println((speedModel ? "Modèle vitesse (K=" + k + ")" : "Modèle autres (K=" + k + ")") + " entraîné.");
        return knn;
    }

    private String predictWithModel(KNearest model, File imageFile, boolean speedModel,
                                    ArrayList<String> labels, int k) {
        Mat feature = extractFeature(imageFile, speedModel);
        return predictFromFeature(model, feature, labels, k);
    }

    private String predictWithModelFromMat(KNearest model, Mat signCrop, boolean speedModel,
                                           ArrayList<String> labels, int k) {
        Mat feature = speedModel
                ? SpeedLimitFeatures.toHogFromMat(signCrop)
                : SignFeatures.toHogFromMat(signCrop);
        return predictFromFeature(model, feature, labels, k);
    }

    private String predictFromFeature(KNearest model, Mat feature, ArrayList<String> labels) {
        return predictFromFeature(model, feature, labels, 1);
    }

    private String predictFromFeature(KNearest model, Mat feature, ArrayList<String> labels, int k) {
        if (feature.empty()) {
            return "IMAGE_NON_TRAITEE";
        }

        Mat results = new Mat();
        float prediction = model.findNearest(feature, k, results);
        int id = Math.round(prediction);

        if (id >= 0 && id < labels.size()) {
            return labels.get(id);
        }
        return "INCONNU";
    }

    private Mat extractFeature(File file, boolean speedModel) {
        return speedModel ? SpeedLimitFeatures.toHog(file) : SignFeatures.toHog(file);
    }

    private static void loadLabels(String trainPath, boolean speedOnly,
                                   ArrayList<String> idToLabel,
                                   HashMap<String, Integer> labelToId) {
        File trainFolder = new File(trainPath);
        File[] folders = trainFolder.listFiles();
        if (folders == null) {
            return;
        }

        ArrayList<String> names = new ArrayList<String>();
        for (File folder : folders) {
            if (!folder.isDirectory()) {
                continue;
            }
            boolean isSpeed = folder.getName().startsWith("Speed_limit_");
            if (speedOnly == isSpeed) {
                names.add(folder.getName());
            }
        }

        Collections.sort(names);
        for (int i = 0; i < names.size(); i++) {
            idToLabel.add(names.get(i));
            labelToId.put(names.get(i), i);
        }
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
