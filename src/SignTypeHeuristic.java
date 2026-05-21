import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Distingue panneaux de vitesse vs autres, puis corrige certaines confusions
 * fréquentes sur photos réelles (ex. No_passing vs Stop).
 */
public class SignTypeHeuristic {

    public static class ColorProfile {
        public final double redRatio;
        public final double blackRatio;
        public final double whiteRatio;

        ColorProfile(double redRatio, double blackRatio, double whiteRatio) {
            this.redRatio = redRatio;
            this.blackRatio = blackRatio;
            this.whiteRatio = whiteRatio;
        }
    }

    public static String detectType(Mat signImage) {
        return detectType(signImage, true);
    }

    public static String detectType(Mat signImage, boolean logProfile) {
        ColorProfile profile = analyzeCenter(signImage);
        if (profile == null) {
            return "UNKNOWN";
        }

        if (logProfile) {
            System.out.println("Heuristique type - redRatio=" + profile.redRatio
                    + " blackRatio=" + profile.blackRatio);
        }

        /*
         * Limitation de vitesse : chiffres noirs au centre, peu de rouge (le rouge reste sur le bord).
         * On teste AVANT le seuil rouge bas (0.025) qui envoyait les "30" vers NON_SPEED.
         */
        if (profile.blackRatio > 0.08 && profile.redRatio < 0.12) {
            return "SPEED";
        }

        if (profile.blackRatio > 0.06 && profile.blackRatio > profile.redRatio * 4.0) {
            return "SPEED";
        }

        // Symbole rouge au centre (dépassement, danger, etc.)
        if (profile.redRatio > 0.08) {
            return "NON_SPEED";
        }

        if (profile.blackRatio > 0.015) {
            return "SPEED";
        }

        return "UNKNOWN";
    }

    /**
     * Panneau rond de limitation de vitesse : chiffres noirs dominants, peu de symbole rouge au centre.
     */
    public static boolean looksLikeSpeedLimitSign(Mat signImage) {
        ColorProfile p = analyzeCenter(signImage);
        if (p == null) {
            return false;
        }
        return p.blackRatio > 0.22 && p.redRatio < 0.20;
    }

    /** Faux positifs de détection : trop de rouge au centre (pas un panneau de vitesse). */
    public static boolean isLikelySpeedLimitCrop(Mat signImage) {
        ColorProfile p = analyzeCenter(signImage);
        if (p == null) {
            return true;
        }
        if (p.redRatio > 0.20) {
            return false;
        }
        return true;
    }

    /**
     * Corrige le label KNN pour les panneaux « autres » sur scène réelle.
     * No_passing : cercle rouge, symboles rouges au centre, peu de texte noir.
     * Stop : beaucoup de lettres noires (STOP) au centre.
     */
    public static String refineOtherLabel(Mat signImage, String knnLabel) {
        return refineOtherLabel(signImage, knnLabel, false);
    }

    public static String refineOtherLabel(Mat signImage, String knnLabel, boolean quiet) {
        ColorProfile p = analyzeCenter(signImage);
        if (p == null) {
            return knnLabel;
        }

        if ("Stop".equals(knnLabel) && p.redRatio > 0.06 && p.blackRatio < 0.12) {
            if (!quiet) {
                System.out.println("  → correction heuristique : Stop → No_passing");
            }
            return "No_passing";
        }

        if ("No_passing".equals(knnLabel) && p.blackRatio > 0.18 && p.redRatio < 0.04) {
            if (!quiet) {
                System.out.println("  → correction heuristique : No_passing → Stop");
            }
            return "Stop";
        }

        /*
         * Interdiction de dépasser : deux véhicules rouges au centre (redRatio élevé).
         * Le panneau poids lourds / tonnage a plutôt un pictogramme sombre sur fond blanc
         * (peu de rouge au centre) — le KNN les confond souvent sur photo réelle.
         */
        if ("Vehicles_over_3_5_metric_tons_prohibited".equals(knnLabel) && looksLikeNoPassing(p)) {
            if (!quiet) {
                System.out.println("  → correction heuristique : Vehicles_over_3_5_metric_tons_prohibited → No_passing");
            }
            return "No_passing";
        }

        if ("No_passing_for_vehicles_over_3_5_metric_tons".equals(knnLabel) && looksLikeNoPassing(p)) {
            if (!quiet) {
                System.out.println("  → correction heuristique : No_passing_for_vehicles_over_3_5_metric_tons → No_passing");
            }
            return "No_passing";
        }

        return knnLabel;
    }

    /**
     * Symboles rouges au centre, peu de noir (pas STOP ni camion noir sur fond blanc).
     * Sur les crops dataset le rouge est souvent faible ; sur scène réelle (p10) il dépasse 0.10.
     */
    static boolean looksLikeNoPassing(ColorProfile p) {
        return p.redRatio > 0.09 && p.blackRatio < 0.14;
    }

    public static ColorProfile analyzeCenter(Mat signImage) {
        if (signImage == null || signImage.empty()) {
            return null;
        }

        Mat resized = new Mat();
        Imgproc.resize(signImage, resized, new Size(128, 128));

        Rect centerRoi = new Rect(30, 30, 68, 68);
        Mat center = new Mat(resized, centerRoi);

        Mat hsv = new Mat();
        Imgproc.cvtColor(center, hsv, Imgproc.COLOR_BGR2HSV);

        Mat redMask1 = new Mat();
        Mat redMask2 = new Mat();
        Mat redMask = new Mat();
        Core.inRange(hsv, new Scalar(0, 80, 50), new Scalar(10, 255, 255), redMask1);
        Core.inRange(hsv, new Scalar(170, 80, 50), new Scalar(180, 255, 255), redMask2);
        Core.bitwise_or(redMask1, redMask2, redMask);

        Mat blackMask = new Mat();
        Core.inRange(hsv, new Scalar(0, 0, 0), new Scalar(180, 120, 120), blackMask);

        Mat whiteMask = new Mat();
        Core.inRange(hsv, new Scalar(0, 0, 180), new Scalar(180, 40, 255), whiteMask);

        double total = center.rows() * center.cols();
        double redRatio = Core.countNonZero(redMask) / total;
        double blackRatio = Core.countNonZero(blackMask) / total;
        double whiteRatio = Core.countNonZero(whiteMask) / total;

        return new ColorProfile(redRatio, blackRatio, whiteRatio);
    }
}
