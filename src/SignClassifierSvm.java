import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Même rôle que {@link SignClassifier}, mais avec des SVM (OpenCV ml).
 * Le KNN d'origine reste intact pour comparer les deux méthodes.
 *
 * Un petit KNN vitesse (K=5) est aussi entraîné uniquement pour
 * {@link SpeedLimitRefinement} sur les photos (voisins proches).
 */
public class SignClassifierSvm {

    private static final int K_REFINE = 5;
    private static final int MAX_IMAGES_PER_CLASS = 200;

    private final ArrayList<String> speedLabels = new ArrayList<String>();
    private final HashMap<String, Integer> speedLabelToId = new HashMap<String, Integer>();

    private final ArrayList<String> otherLabels = new ArrayList<String>();
    private final HashMap<String, Integer> otherLabelToId = new HashMap<String, Integer>();

    private SVM speedSvm;
    private SVM otherSvm;
    private KNearest speedKnnRefine;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String trainPath = "dataset_filtered/train";
        String testPath = "dataset_filtered/test";

        SignClassifierSvm classifier = new SignClassifierSvm();
        classifier.prepare(trainPath);

        System.out.println("=== Entraînement SVM ===");
        classifier.train(trainPath);

        System.out.println("\n=== Évaluation SVM sur le test ===");
        classifier.evaluate(testPath);

        System.out.println("\nTerminé. Rapports SVM dans : results/");
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

        speedSvm = trainSvm(speedSamples, true, "Modèle vitesse SVM");
        otherSvm = trainSvm(otherSamples, false, "Modèle autres SVM");
        speedKnnRefine = trainKnnRefine(speedSamples, true);

        if (speedSvm == null || otherSvm == null) {
            throw new IllegalStateException("Échec d'entraînement SVM — vérifiez dataset_filtered/train.");
        }
    }

    public KNearest getSpeedKnnForRefinement() {
        return speedKnnRefine;
    }

    public String predict(File imageFile, String signType) {
        return predict(imageFile, null, signType);
    }

    public String predict(Mat signCrop, String signType) {
        return predict(signCrop, signType, false);
    }

    public String predict(Mat signCrop, String signType, boolean forVideo) {
        return predict(signCrop, signType, forVideo, 0.0);
    }

    public String predict(Mat signCrop, String signType, boolean forVideo, double signAreaRatio) {
        if (signCrop == null || signCrop.empty()) {
            return "IMAGE_NON_TRAITEE";
        }
        if ("SPEED".equals(signType)) {
            String label = predictFromMat(speedSvm, signCrop, true, speedLabels);
            return SpeedLimitRefinement.refine(signCrop, label, speedKnnRefine, null, speedLabels,
                    forVideo, signAreaRatio);
        }
        String label = predictFromMat(otherSvm, signCrop, false, otherLabels);
        return SignTypeHeuristic.refineOtherLabel(signCrop, label, forVideo);
    }

    public String predict(File imageFile, Mat signCrop, String signType) {
        if ("SPEED".equals(signType)) {
            String label = predictFromFile(speedSvm, imageFile, true, speedLabels);
            if (signCrop != null && !signCrop.empty()) {
                label = SpeedLimitRefinement.refine(signCrop, label, speedKnnRefine, imageFile, speedLabels);
            }
            return label;
        }

        String label = predictFromFile(otherSvm, imageFile, false, otherLabels);

        if (signCrop != null && !signCrop.empty()) {
            label = SignTypeHeuristic.refineOtherLabel(signCrop, label);
        }

        return label;
    }

    public void evaluate(String testPath) {
        File resultsDir = new File("results");
        resultsDir.mkdirs();

        evaluateSplit(testPath, true, speedSvm, speedLabels,
                new File(resultsDir, "speed_limit_test_svm.csv"), "Limitation de vitesse (SVM)");

        evaluateSplit(testPath, false, otherSvm, otherLabels,
                new File(resultsDir, "other_signs_test_svm.csv"), "Autres panneaux (SVM)");
    }

    private void evaluateSplit(String testPath, boolean speedOnly, SVM model,
                               ArrayList<String> labels, File reportFile, String title) {
        ArrayList<Sample> samples = loadSamples(testPath, speedOnly,
                speedOnly ? speedLabelToId : otherLabelToId, Integer.MAX_VALUE);

        int total = 0;
        int correct = 0;
        Map<String, Integer> totalByClass = new TreeMap<String, Integer>();
        Map<String, Integer> correctByClass = new TreeMap<String, Integer>();

        try (PrintWriter writer = new PrintWriter(reportFile)) {
            writer.println("filename,true_label,predicted_label,result");

            for (Sample sample : samples) {
                String predicted = predictFromFile(model, sample.file, speedOnly, labels);
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

    private SVM trainSvm(ArrayList<Sample> samples, boolean speedModel, String logName) {
        if (samples.isEmpty()) {
            return null;
        }

        Mat first = extractFeature(samples.get(0).file, speedModel);
        if (first.empty()) {
            return null;
        }

        int featureLength = first.cols();
        Mat trainData = new Mat(samples.size(), featureLength, CvType.CV_32F);
        Mat trainLabels = new Mat(samples.size(), 1, CvType.CV_32S);
        first.release();

        for (int i = 0; i < samples.size(); i++) {
            Mat feature = extractFeature(samples.get(i).file, speedModel);
            if (!feature.empty()) {
                feature.copyTo(trainData.row(i));
                trainLabels.put(i, 0, samples.get(i).labelId);
            }
            feature.release();
        }

        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.RBF);
        svm.setGamma(0.005);
        svm.setC(50);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 1000, 1e-6));

        if (!svm.train(trainData, Ml.ROW_SAMPLE, trainLabels)) {
            return null;
        }

        System.out.println(logName + " entraîné.");
        return svm;
    }

    /** KNN auxiliaire (même features) pour le raffinement heuristique des vitesses. */
    private KNearest trainKnnRefine(ArrayList<Sample> samples, boolean speedModel) {
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
        knn.setDefaultK(K_REFINE);
        knn.setIsClassifier(true);

        if (!knn.train(trainData, Ml.ROW_SAMPLE, trainLabels)) {
            return null;
        }

        System.out.println("KNN auxiliaire vitesse (K=" + K_REFINE + ", raffinement) entraîné.");
        return knn;
    }

    private String predictFromFile(SVM model, File imageFile, boolean speedModel, ArrayList<String> labels) {
        Mat feature = extractFeature(imageFile, speedModel);
        return predictFromFeature(model, feature, labels);
    }

    private String predictFromMat(SVM model, Mat signCrop, boolean speedModel, ArrayList<String> labels) {
        Mat feature = speedModel
                ? SpeedLimitFeatures.toHogFromMat(signCrop)
                : SignFeatures.toHogFromMat(signCrop);
        return predictFromFeature(model, feature, labels);
    }

    private String predictFromFeature(SVM model, Mat feature, ArrayList<String> labels) {
        if (model == null || feature.empty()) {
            return "IMAGE_NON_TRAITEE";
        }

        float prediction = model.predict(feature);
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
