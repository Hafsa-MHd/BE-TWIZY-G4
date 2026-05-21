import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class DatasetFilter {

    // Classes qu'on exclut car elles ne correspondent pas directement à notre étude des panneaux.
    private static final Set<String> CLASSES_A_EXCLURE = new HashSet<String>(Arrays.asList(
            "green-lights",
            "red-lights",
            "yellow-lights",
            "traffic-lights",
            "turn-left",
            "turn-right"
    ));

    // Seuil minimal de couleur détectée.
    // Si trop d'images sont refusées, on pourra diminuer cette valeur.
    private static final double SEUIL_COULEUR_UTILE = 0.001;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        traiterDossier("dataset/train", "dataset_filtered/train");
        traiterDossier("dataset/valid", "dataset_filtered/valid");
        traiterDossier("dataset/test", "dataset_filtered/test");

        System.out.println("\nTraitement terminé.");
        System.out.println("Regarde le dossier : dataset_filtered");
    }

    public static void traiterDossier(String inputFolderPath, String outputFolderPath) {
        File inputFolder = new File(inputFolderPath);
        File csvFile = new File(inputFolder, "_classes.csv");

        if (!inputFolder.exists()) {
            System.out.println("Dossier introuvable : " + inputFolderPath);
            return;
        }

        if (!csvFile.exists()) {
            System.out.println("CSV introuvable : " + csvFile.getAbsolutePath());
            return;
        }

        File outputFolder = new File(outputFolderPath);
        outputFolder.mkdirs();

        int total = 0;
        int sansLabel = 0;
        int plusieursLabels = 0;
        int classesExclues = 0;
        int imagesIllisibles = 0;
        int imagesSansCouleurUtile = 0;
        int imagesRetenues = 0;

        File reportFile = new File(outputFolder, "rapport_filtrage.csv");

        try {
            BufferedReader reader = new BufferedReader(new FileReader(csvFile));
            PrintWriter report = new PrintWriter(reportFile);

            report.println("filename,label,red_ratio,yellow_ratio,blue_ratio,green_ratio,decision");

            String headerLine = reader.readLine();

            if (headerLine == null) {
                System.out.println("CSV vide : " + csvFile.getAbsolutePath());
                reader.close();
                report.close();
                return;
            }

            String[] headers = headerLine.split(",");

            String line;

            while ((line = reader.readLine()) != null) {
                total++;

                String[] values = line.split(",");

                if (values.length < 2) {
                    sansLabel++;
                    continue;
                }

                String filename = values[0].trim();
                String label = trouverLabel(headers, values);
                int nombreLabels = compterLabelsActifs(values);

                if (nombreLabels == 0) {
                    sansLabel++;
                    report.println(filename + ",NONE,0,0,0,0,REFUSE_SANS_LABEL");
                    continue;
                }

                if (nombreLabels > 1) {
                    plusieursLabels++;
                    report.println(filename + ",MULTIPLE,0,0,0,0,REFUSE_PLUSIEURS_LABELS");
                    continue;
                }

                if (label == null) {
                    sansLabel++;
                    report.println(filename + ",UNKNOWN,0,0,0,0,REFUSE_LABEL_INCONNU");
                    continue;
                }

                if (CLASSES_A_EXCLURE.contains(label)) {
                    classesExclues++;
                    report.println(filename + "," + label + ",0,0,0,0,REFUSE_CLASSE_EXCLUE");
                    continue;
                }

                File imageFile = new File(inputFolder, filename);

                if (!imageFile.exists()) {
                    imagesIllisibles++;
                    report.println(filename + "," + label + ",0,0,0,0,REFUSE_IMAGE_INTROUVABLE");
                    continue;
                }

                Mat image = Imgcodecs.imread(imageFile.getAbsolutePath());

                if (image.empty()) {
                    imagesIllisibles++;
                    report.println(filename + "," + label + ",0,0,0,0,REFUSE_IMAGE_ILLISIBLE");
                    continue;
                }

                ColorStats stats = analyserCouleurs(image);

                boolean couleurUtile =
                        stats.redRatio >= SEUIL_COULEUR_UTILE ||
                        stats.yellowRatio >= SEUIL_COULEUR_UTILE ||
                        stats.blueRatio >= SEUIL_COULEUR_UTILE ||
                        stats.greenRatio >= SEUIL_COULEUR_UTILE;

                if (!couleurUtile) {
                    imagesSansCouleurUtile++;
                    report.println(filename + "," + label + ","
                            + stats.redRatio + ","
                            + stats.yellowRatio + ","
                            + stats.blueRatio + ","
                            + stats.greenRatio + ",REFUSE_COULEUR_FAIBLE");
                    continue;
                }

                File classFolder = new File(outputFolder, label);
                classFolder.mkdirs();

                File destination = new File(classFolder, filename);

                Files.copy(
                        imageFile.toPath(),
                        destination.toPath(),
                        StandardCopyOption.REPLACE_EXISTING
                );

                imagesRetenues++;

                report.println(filename + "," + label + ","
                        + stats.redRatio + ","
                        + stats.yellowRatio + ","
                        + stats.blueRatio + ","
                        + stats.greenRatio + ",RETENUE");
            }

            reader.close();
            report.close();

        } catch (Exception e) {
            System.out.println("Erreur pendant le traitement de : " + inputFolderPath);
            e.printStackTrace();
            return;
        }

        System.out.println("\n==============================");
        System.out.println("Filtrage du dossier : " + inputFolderPath);
        System.out.println("==============================");
        System.out.println("Images dans le CSV : " + total);
        System.out.println("Images sans label : " + sansLabel);
        System.out.println("Images avec plusieurs labels : " + plusieursLabels);
        System.out.println("Classes exclues : " + classesExclues);
        System.out.println("Images illisibles/introuvables : " + imagesIllisibles);
        System.out.println("Images sans couleur utile : " + imagesSansCouleurUtile);
        System.out.println("Images retenues : " + imagesRetenues);
        System.out.println("Rapport créé : " + reportFile.getAbsolutePath());
    }

    public static String trouverLabel(String[] headers, String[] values) {
        int nombreLabelsActifs = 0;
        String labelTrouve = null;

        for (int i = 1; i < values.length && i < headers.length; i++) {
            if (values[i].trim().equals("1")) {
                nombreLabelsActifs++;
                labelTrouve = headers[i].trim();
            }
        }

        if (nombreLabelsActifs == 1) {
            return labelTrouve;
        }

        return null;
    }

    public static int compterLabelsActifs(String[] values) {
        int compteur = 0;

        for (int i = 1; i < values.length; i++) {
            if (values[i].trim().equals("1")) {
                compteur++;
            }
        }

        return compteur;
    }

    public static ColorStats analyserCouleurs(Mat image) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(image, hsv, Imgproc.COLOR_BGR2HSV);

        Mat redMask1 = new Mat();
        Mat redMask2 = new Mat();
        Mat redMask = new Mat();

        Core.inRange(hsv, new Scalar(0, 110, 50), new Scalar(6, 255, 255), redMask1);
        Core.inRange(hsv, new Scalar(170, 110, 50), new Scalar(180, 255, 255), redMask2);
        Core.bitwise_or(redMask1, redMask2, redMask);

        Mat yellowMask = new Mat();
        Core.inRange(hsv, new Scalar(20, 80, 80), new Scalar(35, 255, 255), yellowMask);

        Mat blueMask = new Mat();
        Core.inRange(hsv, new Scalar(90, 80, 50), new Scalar(130, 255, 255), blueMask);

        Mat greenMask = new Mat();
        Core.inRange(hsv, new Scalar(40, 80, 50), new Scalar(85, 255, 255), greenMask);

        double totalPixels = image.rows() * image.cols();

        double redRatio = Core.countNonZero(redMask) / totalPixels;
        double yellowRatio = Core.countNonZero(yellowMask) / totalPixels;
        double blueRatio = Core.countNonZero(blueMask) / totalPixels;
        double greenRatio = Core.countNonZero(greenMask) / totalPixels;

        return new ColorStats(redRatio, yellowRatio, blueRatio, greenRatio);
    }

    static class ColorStats {
        double redRatio;
        double yellowRatio;
        double blueRatio;
        double greenRatio;

        ColorStats(double redRatio, double yellowRatio, double blueRatio, double greenRatio) {
            this.redRatio = redRatio;
            this.yellowRatio = yellowRatio;
            this.blueRatio = blueRatio;
            this.greenRatio = greenRatio;
        }
    }
}
