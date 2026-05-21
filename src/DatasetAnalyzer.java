import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Map;
import java.util.TreeMap;

public class DatasetAnalyzer {

    public static void main(String[] args) {
        analyserDossier("dataset/train");
        analyserDossier("dataset/valid");
        analyserDossier("dataset/test");
    }

    public static void analyserDossier(String dossierPath) {
        File dossier = new File(dossierPath);
        File csv = new File(dossier, "_classes.csv");

        if (!dossier.exists()) {
            System.out.println("Dossier introuvable : " + dossierPath);
            return;
        }

        if (!csv.exists()) {
            System.out.println("CSV introuvable : " + csv.getAbsolutePath());
            return;
        }

        Map<String, Integer> compteurClasses = new TreeMap<String, Integer>();

        int totalImages = 0;
        int imagesSansLabel = 0;
        int imagesAvecPlusieursLabels = 0;
        int imagesAvecLabel = 0;
        int imagesIntrouvables = 0;

        try {
            BufferedReader reader = new BufferedReader(new FileReader(csv));

            String headerLine = reader.readLine();

            if (headerLine == null) {
                System.out.println("CSV vide : " + csv.getAbsolutePath());
                reader.close();
                return;
            }

            String[] headers = headerLine.split(",");

            String line;

            while ((line = reader.readLine()) != null) {
                totalImages++;

                String[] values = line.split(",");

                if (values.length < 2) {
                    imagesSansLabel++;
                    continue;
                }

                String filename = values[0].trim();
                File imageFile = new File(dossier, filename);

                if (!imageFile.exists()) {
                    imagesIntrouvables++;
                }

                String label = trouverLabel(headers, values);

                if (label == null) {
                    int nombreLabels = compterLabelsActifs(values);

                    if (nombreLabels == 0) {
                        imagesSansLabel++;
                    } else {
                        imagesAvecPlusieursLabels++;
                    }

                    continue;
                }

                imagesAvecLabel++;
                ajouterClasse(compteurClasses, label);
            }

            reader.close();

        } catch (Exception e) {
            System.out.println("Erreur pendant la lecture de : " + csv.getAbsolutePath());
            e.printStackTrace();
            return;
        }

        System.out.println("\n==============================");
        System.out.println("Analyse du dossier : " + dossierPath);
        System.out.println("==============================");
        System.out.println("Images listées dans le CSV : " + totalImages);
        System.out.println("Images avec un seul label : " + imagesAvecLabel);
        System.out.println("Images sans label : " + imagesSansLabel);
        System.out.println("Images avec plusieurs labels : " + imagesAvecPlusieursLabels);
        System.out.println("Images introuvables : " + imagesIntrouvables);

        System.out.println("\nRépartition des classes :");
        for (String classe : compteurClasses.keySet()) {
            System.out.println(classe + " : " + compteurClasses.get(classe));
        }
    }

    public static String trouverLabel(String[] headers, String[] values) {
        int nombreLabelsActifs = 0;
        String labelTrouve = null;

        for (int i = 1; i < values.length && i < headers.length; i++) {
            String value = values[i].trim();

            if (value.equals("1")) {
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

    public static void ajouterClasse(Map<String, Integer> compteurClasses, String label) {
        if (!compteurClasses.containsKey(label)) {
            compteurClasses.put(label, 0);
        }

        compteurClasses.put(label, compteurClasses.get(label) + 1);
    }
}