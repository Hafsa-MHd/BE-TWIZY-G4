import org.opencv.core.Core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

/**
 * Entraîne KNN et SVM, évalue les deux sur dataset_filtered/test,
 * affiche un tableau comparatif pour le rapport / l'enseignant.
 *
 * Rapports détaillés :
 * - results/speed_limit_test.csv          (KNN)
 * - results/speed_limit_test_svm.csv      (SVM)
 * - results/other_signs_test.csv          (KNN)
 * - results/other_signs_test_svm.csv      (SVM)
 */
public class SignClassifierCompare {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String trainPath = "dataset_filtered/train";
        String testPath = "dataset_filtered/test";

        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║     Comparaison KNN vs SVM (même HOG, même test)     ║");
        System.out.println("╚══════════════════════════════════════════════════════╝\n");

        System.out.println("--- 1/2 Entraînement et test KNN ---");
        SignClassifier knn = new SignClassifier();
        knn.prepare(trainPath);
        knn.train(trainPath);
        knn.evaluate(testPath);

        System.out.println("\n--- 2/2 Entraînement et test SVM ---");
        SignClassifierSvm svm = new SignClassifierSvm();
        svm.prepare(trainPath);
        svm.train(trainPath);
        svm.evaluate(testPath);

        System.out.println("\n╔══════════════════════════════════════════════════════╗");
        System.out.println("║              RÉSUMÉ GLOBAL (précision %)             ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");

        printSummary("Limitation de vitesse",
                "results/speed_limit_test.csv",
                "results/speed_limit_test_svm.csv");

        printSummary("Autres panneaux",
                "results/other_signs_test.csv",
                "results/other_signs_test_svm.csv");

        System.out.println("\nPour des images réelles : lancez SceneRecognition puis SceneRecognitionSvm.");
        System.out.println("Pour la vidéo            : VideoRecognition puis VideoRecognitionSvm.");
    }

    private static void printSummary(String title, String knnCsv, String svmCsv) {
        double knnAcc = accuracyFromCsv(new File(knnCsv));
        double svmAcc = accuracyFromCsv(new File(svmCsv));

        System.out.println("\n" + title + " :");
        System.out.printf("  KNN : %.1f %%  (%s)%n", knnAcc, knnCsv);
        System.out.printf("  SVM : %.1f %%  (%s)%n", svmAcc, svmCsv);

        if (svmAcc > knnAcc) {
            System.out.printf("  → SVM gagne de %.1f point(s) sur ce jeu de test.%n", svmAcc - knnAcc);
        } else if (knnAcc > svmAcc) {
            System.out.printf("  → KNN gagne de %.1f point(s) sur ce jeu de test.%n", knnAcc - svmAcc);
        } else {
            System.out.println("  → Égalité sur ce jeu de test.");
        }
    }

    private static double accuracyFromCsv(File csv) {
        if (!csv.isFile()) {
            return 0.0;
        }

        int total = 0;
        int ok = 0;

        try (BufferedReader reader = new BufferedReader(new FileReader(csv))) {
            String line = reader.readLine();
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 4) {
                    total++;
                    if ("OK".equals(parts[3].trim())) {
                        ok++;
                    }
                }
            }
        } catch (Exception e) {
            return 0.0;
        }

        return total > 0 ? (100.0 * ok) / total : 0.0;
    }
}
