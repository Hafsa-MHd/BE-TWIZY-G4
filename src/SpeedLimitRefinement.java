import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Corrige le KNN en relisant les chiffres sur le masque noir du panneau.
 */
public class SpeedLimitRefinement {

    private static void logSpeedCorrection(boolean videoMode, String message) {
        if (!videoMode) {
            System.out.println(message);
        }
    }

    public static String refine(Mat signImage, String knnLabel) {
        return refine(signImage, knnLabel, null, null, null);
    }

    public static String refine(Mat signImage, String knnLabel, KNearest speedModel,
                                File imageFile, List<String> speedLabels) {
        return refine(signImage, knnLabel, speedModel, imageFile, speedLabels, false);
    }

    /**
     * @param videoMode si true : garde le KNN, désactive les lectures OCR instables (vidéo).
     */
    public static String refine(Mat signImage, String knnLabel, KNearest speedModel,
                                File imageFile, List<String> speedLabels, boolean videoMode) {
        return refine(signImage, knnLabel, speedModel, imageFile, speedLabels, videoMode, 0.0);
    }

    /**
     * @param signAreaRatio part de l'image occupée par le panneau (0 si inconnu).
     */
    public static String refine(Mat signImage, String knnLabel, KNearest speedModel,
                                File imageFile, List<String> speedLabels, boolean videoMode,
                                double signAreaRatio) {
        if (signImage == null || signImage.empty() || knnLabel == null) {
            return knnLabel;
        }
        if (!knnLabel.startsWith("Speed_limit_")) {
            return knnLabel;
        }

        String refined = knnLabel;
        String reason = null;

        Mat mask = SpeedLimitFeatures.extractDigitMask(signImage);

        // 0) Panneau à trois chiffres type 110 (prioritaire, p5.jpg)
        if (!mask.empty() && speedLabels != null
                && speedLabels.contains("Speed_limit_110_km_h")) {
            if (looksLike110Layout(mask)) {
                int knnSpeed = parseSpeed(knnLabel);
                if (knnSpeed != 110) {
                    logSpeedCorrection(videoMode, "  → correction vitesse : " + knnLabel
                            + " → Speed_limit_110_km_h (layout 110)");
                    return "Speed_limit_110_km_h";
                }
            } else {
                String fromProfile = fix110FromSceneProfile(signImage, knnLabel, mask);
                if (!fromProfile.equals(knnLabel)) {
                    logSpeedCorrection(videoMode, "  → correction vitesse : " + knnLabel + " → "
                            + fromProfile + " (profil scène 110)");
                    return fromProfile;
                }
            }
        }

        // 0a) 100 → 110 : chiffre du milieu = 1 (p3.jpg et similaires)
        if ("Speed_limit_100_km_h".equals(knnLabel) && !mask.empty()) {
            String fix110 = fix100vs110(signImage, knnLabel, speedModel, speedLabels, mask);
            if (!fix110.equals(knnLabel)) {
                logSpeedCorrection(videoMode, "  → correction vitesse : " + knnLabel + " → " + fix110
                        + " (100↔110)");
                return fix110;
            }
        }

        // 0a2) 40 → 110 : trois chiffres (p5.jpg — le KNN confond souvent 40 et 110)
        if ("Speed_limit_40_km_h".equals(knnLabel) && !mask.empty()) {
            String fix110 = fix40vs110(signImage, knnLabel, speedModel, speedLabels, mask);
            if (!fix110.equals(knnLabel)) {
                logSpeedCorrection(videoMode, "  → correction vitesse : " + knnLabel + " → " + fix110
                        + " (40↔110)");
                return fix110;
            }
        }

        // 0b) 30 → 90 : voisins KNN proches (photos seulement)
        if (!videoMode && speedModel != null && speedLabels != null) {
            String pairFix = fixClosePair(signImage, knnLabel, speedModel, speedLabels,
                    "Speed_limit_30_km_h", "Speed_limit_90_km_h", 1.70f);
            if (!pairFix.equals(knnLabel)) {
                logSpeedCorrection(videoMode, "  → correction vitesse : " + knnLabel + " → " + pairFix
                        + " (30↔90 : voisin KNN proche)");
                return pairFix;
            }
        }

        if (mask.empty()) {
            return knnLabel;
        }

        if (!videoMode) {
            // 1) Lecture directe des chiffres (instable sur vidéo : 70→30, 10→20…)
            if (speedLabels != null) {
                String read = readSpeedFromMask(mask, speedLabels);
                if (read != null && !read.equals(knnLabel)
                        && shouldTrustDigitReadOverKnn(knnLabel, read, mask, speedLabels)) {
                    refined = read;
                    reason = "chiffres lus sur le panneau";
                }
            }

            // 2) Règles selon nombre de bandes
            if (refined.equals(knnLabel)) {
                List<int[]> bands = getDigitBands(mask);
                if (bands.size() >= 3) {
                    Character mid = readDigitInBand(mask, bands.get(1));
                    if (mid == null) {
                        mid = detectMiddleDigitByThirds(mask);
                    }
                    refined = applyRulesThreeDigit(knnLabel, mid);
                    if (!refined.equals(knnLabel) && mid != null) {
                        reason = "milieu ≈ " + mid;
                    }
                } else {
                    Character first = readFirstDigit(mask, bands);
                    String fromFirst = applyRulesTwoDigit(knnLabel, first);
                    if (!fromFirst.equals(knnLabel) && first != null
                            && shouldTrustDigitReadOverKnn(knnLabel, fromFirst, mask, speedLabels)) {
                        refined = fromFirst;
                        reason = "1er chiffre ≈ " + first;
                    }
                }
            }

            // 3) Voisins KNN proches
            if (refined.equals(knnLabel) && speedModel != null && speedLabels != null) {
                String alt = refineFromNeighbors(signImage, knnLabel, speedModel, speedLabels, mask);
                if (!alt.equals(knnLabel)) {
                    refined = alt;
                    reason = "voisins KNN";
                }
            }

            // Dernier recours : 30↔90 très fréquent sur photos réelles
            if ("Speed_limit_30_km_h".equals(refined) && speedLabels != null
                    && speedLabels.contains("Speed_limit_90_km_h")) {
                Character nine = looksLikeNine(mask);
                if (nine != null && nine == '9') {
                    refined = "Speed_limit_90_km_h";
                    reason = "forme du 9 détectée";
                }
            }
        }

        if (videoMode) {
            refined = refineForVideo(signImage, refined, mask, speedLabels, signAreaRatio);
        } else if (!refined.equals(knnLabel)) {
            System.out.println("  → correction vitesse : " + knnLabel + " → " + refined
                    + (reason != null ? " (" + reason + ")" : ""));
        }

        return refined;
    }

    /**
     * Corrections stables pour la vidéo : lecture du masque uniquement (pas de voisins KNN ni OCR agressif).
     */
    private static String refineForVideo(Mat signImage, String label, Mat mask,
                                         List<String> speedLabels, double signAreaRatio) {
        if (label == null || !label.startsWith("Speed_limit_") || speedLabels == null) {
            return label;
        }
        if (mask.empty()) {
            mask = SpeedLimitFeatures.extractDigitMask(signImage);
        }
        if (mask.empty()) {
            return label;
        }

        int speed = parseSpeed(label);

        // 40 / 30 → 90 : SVM et KNN confondent souvent le « 9 » avec « 4 » ou « 3 »
        if (speed == 40 || speed == 30) {
            Character nine = looksLikeNine(mask);
            if (nine != null && nine == '9') {
                if (speedLabels.contains("Speed_limit_90_km_h")) {
                    return "Speed_limit_90_km_h";
                }
            }
            String read = readSpeedFromMask(mask, speedLabels);
            if ("Speed_limit_90_km_h".equals(read)) {
                return read;
            }
            if (speed == 40 && countHoles(new Mat(mask, new Rect(0, 0, Math.max(mask.cols() / 2, 1), mask.rows()))) >= 1
                    && speedLabels.contains("Speed_limit_90_km_h")) {
                return "Speed_limit_90_km_h";
            }
        }

        // 60 / 80 → 90 si le masque lit 90 avec confiance
        if (speed == 60 || speed == 80) {
            String read = readSpeedFromMask(mask, speedLabels);
            if ("Speed_limit_90_km_h".equals(read)
                    && isDigitReadConfident(mask, read, speedLabels)) {
                return read;
            }
        }

        // 70 → 50 : SVM dit 70 alors que les chiffres lisent 50 (fin de video1.avi)
        if ("Speed_limit_70_km_h".equals(label) && speedLabels.contains("Speed_limit_50_km_h")) {
            String read = readSpeedFromMask(mask, speedLabels);
            if ("Speed_limit_50_km_h".equals(read)
                    && isDigitReadConfident(mask, read, speedLabels)) {
                return read;
            }
            if (signAreaRatio >= 0.006) {
                List<int[]> bands = getDigitBands(mask);
                if (!bands.isEmpty()) {
                    if (bands.size() > 2) {
                        bands = keepTwoWidestBands(bands);
                    }
                    Character first = readDigitInBand(mask, bands.get(0));
                    if (first != null && first == '5') {
                        return "Speed_limit_50_km_h";
                    }
                }
            }
        }

        return label;
    }

    /**
     * Si le KNN dit A mais B est un voisin presque aussi proche, on prend B.
     * Utilisé pour 30↔90 sur photos externes où le HOG se trompe souvent.
     */
    private static String fixClosePair(Mat signImage, String knnLabel, KNearest model,
                                       List<String> speedLabels, String labelA, String labelB,
                                       float maxRatio) {
        if (!labelA.equals(knnLabel)) {
            return knnLabel;
        }

        int idA = speedLabels.indexOf(labelA);
        int idB = speedLabels.indexOf(labelB);
        if (idA < 0 || idB < 0) {
            return knnLabel;
        }

        Mat feature = SpeedLimitFeatures.toHogFromMat(signImage);
        if (feature.empty()) {
            return knnLabel;
        }

        Mat neighborResponses = new Mat();
        Mat dist = new Mat();
        model.findNearest(feature, 7, new Mat(), neighborResponses, dist);

        float distA = Float.MAX_VALUE;
        float distB = Float.MAX_VALUE;

        for (int i = 0; i < neighborResponses.rows(); i++) {
            int nid = (int) neighborResponses.get(i, 0)[0];
            float d = (float) dist.get(i, 0)[0];
            if (nid == idA) {
                distA = Math.min(distA, d);
            }
            if (nid == idB) {
                distB = Math.min(distB, d);
            }
        }

        if (distB < Float.MAX_VALUE && distB <= distA * maxRatio) {
            return labelB;
        }

        return knnLabel;
    }

    private static String fix100vs110(Mat signImage, String knnLabel, KNearest model,
                                      List<String> speedLabels, Mat mask) {
        if (speedLabels == null || !speedLabels.contains("Speed_limit_110_km_h")) {
            return knnLabel;
        }

        String read = readSpeedFromMask(mask, speedLabels);
        if ("Speed_limit_110_km_h".equals(read)) {
            return read;
        }

        Character mid = detectMiddleDigitByThirds(mask);
        if (mid != null && mid == '1') {
            return "Speed_limit_110_km_h";
        }

        if (looksLikeOneInMiddle(mask)) {
            return "Speed_limit_110_km_h";
        }

        if (model != null) {
            String fromNeighbor = fixClosePair(signImage, knnLabel, model, speedLabels,
                    "Speed_limit_100_km_h", "Speed_limit_110_km_h", 1.65f);
            if (!fromNeighbor.equals(knnLabel) && (mid == null || mid != '0')) {
                return fromNeighbor;
            }
        }

        return knnLabel;
    }

    /** KNN dit 40 alors que le panneau affiche 110 (trois chiffres). */
    /**
     * Sur photo réelle : beaucoup d'encre noire (3 chiffres) + KNN à 2 chiffres (40, 90…).
     * Le voisin KNN 110 est souvent absent ; on utilise le masque et le profil couleur.
     */
    private static String fix110FromSceneProfile(Mat signImage, String knnLabel, Mat mask) {
        int knnSpeed = parseSpeed(knnLabel);
        if (knnSpeed == 110 || knnSpeed >= 100) {
            return knnLabel;
        }
        if (knnSpeed != 40 && knnSpeed != 90 && knnSpeed != 70 && knnSpeed != 80) {
            return knnLabel;
        }

        SignTypeHeuristic.ColorProfile p = SignTypeHeuristic.analyzeCenter(signImage);
        if (p == null || p.blackRatio < 0.20 || p.redRatio > 0.10) {
            return knnLabel;
        }

        if (countDigitColumnGroups(mask) >= 3) {
            return "Speed_limit_110_km_h";
        }

        if (digitInkWidthRatio(mask) > 0.82 && hasTwoVerticalOnesAndZero(mask)) {
            return "Speed_limit_110_km_h";
        }

        return knnLabel;
    }

    /** Colonnes d'encre séparées (110 → 3, 40 → 2). */
    private static int countDigitColumnGroups(Mat mask) {
        if (mask.empty()) {
            return 0;
        }
        int cols = mask.cols();
        int rows = mask.rows();
        double thresh = rows * 0.07;
        int groups = 0;
        boolean inGroup = false;
        int gap = 0;

        for (int x = 0; x < cols; x++) {
            int count = 0;
            for (int y = 0; y < rows; y++) {
                if (mask.get(y, x)[0] > 127) {
                    count++;
                }
            }
            if (count > thresh) {
                if (!inGroup && gap >= 2) {
                    groups++;
                }
                inGroup = true;
                gap = 0;
            } else {
                if (inGroup) {
                    gap++;
                }
                if (gap >= 3) {
                    inGroup = false;
                }
            }
        }
        if (inGroup) {
            groups++;
        }
        return groups;
    }

    private static double digitInkWidthRatio(Mat mask) {
        Mat points = new Mat();
        Core.findNonZero(mask, points);
        if (points.rows() < 12) {
            return 0;
        }
        double minX = mask.cols();
        double maxX = 0;
        for (int i = 0; i < points.rows(); i++) {
            double x = points.get(i, 0)[0];
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
        }
        return (maxX - minX + 1) / Math.max(mask.cols(), 1);
    }

    private static String fix40vs110(Mat signImage, String knnLabel, KNearest model,
                                     List<String> speedLabels, Mat mask) {
        if (speedLabels == null || !speedLabels.contains("Speed_limit_110_km_h")) {
            return knnLabel;
        }

        String read = readSpeedFromMask(mask, speedLabels);
        if ("Speed_limit_110_km_h".equals(read)) {
            return read;
        }

        if (looksLikeThreeDigit110(mask)) {
            return "Speed_limit_110_km_h";
        }

        if (model != null && looksLike110Layout(mask)) {
            String fromNeighbor = fixClosePair(signImage, knnLabel, model, speedLabels,
                    "Speed_limit_40_km_h", "Speed_limit_110_km_h", 2.0f);
            if (!fromNeighbor.equals(knnLabel)) {
                return fromNeighbor;
            }
        }

        return knnLabel;
    }

    private static boolean hasThreeDigitBands(Mat mask) {
        return getDigitBands(mask).size() >= 3;
    }

    /**
     * 110 = deux chiffres « 1 » étroits + « 0 ». Plus fiable que looksLikeOneInMiddle seul sur p5.
     */
    private static boolean looksLike110Layout(Mat mask) {
        if (countDigitColumnGroups(mask) >= 3) {
            return true;
        }
        if (hasOneOneZeroThirds(mask)) {
            return true;
        }
        if (getDigitBands(mask).size() <= 2 && hasTwoVerticalOnesAndZero(mask)) {
            return true;
        }
        if (countNarrowDigitBands(mask) >= 2) {
            return true;
        }
        if (hasThreeDigitBands(mask)) {
            List<int[]> bands = getDigitBands(mask);
            if (bands.size() > 3) {
                bands = keepThreeLeftmostBands(bands);
            }
            Character mid = readDigitInBand(mask, bands.get(1));
            if (mid == null) {
                mid = detectMiddleDigitByThirds(mask);
            }
            if (mid != null && mid == '1') {
                return true;
            }
        }
        int w = mask.cols();
        int h = mask.rows();
        if (w < 12) {
            return false;
        }
        int third = w / 3;
        Character t2 = analyzeRegion(new Mat(mask, new Rect(third, 0, third, h)));
        Character t3 = analyzeRegion(new Mat(mask, new Rect(2 * third, 0, w - 2 * third, h)));
        return t2 != null && t2 == '1' && (t3 == null || t3 == '0');
    }

    /** Trois pics + motif 1-1-0 (évite de confondre un « 4 » qui crée aussi 3 pics). */
    private static boolean hasOneOneZeroThirds(Mat mask) {
        int w = mask.cols();
        int h = mask.rows();
        if (w < 12) {
            return false;
        }
        int third = w / 3;
        Character t1 = analyzeRegion(new Mat(mask, new Rect(0, 0, third, h)));
        Character t2 = analyzeRegion(new Mat(mask, new Rect(third, 0, third, h)));
        Character t3 = analyzeRegion(new Mat(mask, new Rect(2 * third, 0, w - 2 * third, h)));

        if (t1 != null && t1 == '4') {
            return false;
        }
        if (t2 != null && t2 == '1') {
            return true;
        }
        return t1 != null && t1 == '1' && (t3 == null || t3 == '0');
    }

    private static boolean looksLikeThreeDigit110(Mat mask) {
        return looksLike110Layout(mask);
    }

    /**
     * Nombre de colonnes « pleines » (pics) — 110 en donne 3, 40 en donne 2.
     * Fonctionne même quand getDigitBands ne voit qu'une seule bande (p5.jpg).
     */
    private static int countInkPeaks(Mat mask) {
        int cols = mask.cols();
        int rows = mask.rows();
        if (cols < 8) {
            return 0;
        }
        double thresh = rows * 0.06;
        int[] proj = new int[cols];
        for (int x = 0; x < cols; x++) {
            int count = 0;
            for (int y = 0; y < rows; y++) {
                if (mask.get(y, x)[0] > 127) {
                    count++;
                }
            }
            proj[x] = count;
        }

        int peaks = 0;
        int lastPeakX = -cols;
        int minGap = Math.max(3, cols / 10);

        for (int x = 1; x < cols - 1; x++) {
            if (proj[x] > thresh
                    && proj[x] >= proj[x - 1]
                    && proj[x] >= proj[x + 1]
                    && x - lastPeakX >= minGap) {
                peaks++;
                lastPeakX = x;
            }
        }
        return peaks;
    }

    private static int countNarrowDigitBands(Mat mask) {
        List<int[]> bands = getDigitBands(mask);
        int narrow = 0;
        int maxW = (int) (mask.cols() * 0.24);
        for (int[] b : bands) {
            int bw = b[1] - b[0] + 1;
            if (bw <= maxW && bw >= 2) {
                narrow++;
            }
        }
        return narrow;
    }

    /** Deux traits verticaux (les « 1 ») + zone droite ronde (« 0 »), même si une seule bande (p5). */
    private static boolean hasTwoVerticalOnesAndZero(Mat mask) {
        int w = mask.cols();
        int h = mask.rows();
        if (w < 12) {
            return false;
        }
        int third = w / 3;
        Mat left = new Mat(mask, new Rect(0, 0, third, h));
        Mat mid = new Mat(mask, new Rect(third, 0, third, h));
        Mat right = new Mat(mask, new Rect(2 * third, 0, w - 2 * third, h));

        if (!hasVerticalStrokeInRegion(left) || !hasVerticalStrokeInRegion(mid)) {
            return false;
        }
        Character t3 = analyzeRegion(right);
        if (t3 != null && t3 != '0') {
            return false;
        }
        Character t1 = analyzeRegion(left);
        return t1 == null || t1 != '4';
    }

    private static boolean hasVerticalStrokeInRegion(Mat region) {
        if (region.empty() || region.cols() < 2) {
            return false;
        }
        int h = region.rows();
        double maxCol = 0;
        double sumCol = 0;
        for (int x = 0; x < region.cols(); x++) {
            int count = 0;
            for (int y = 0; y < h; y++) {
                if (region.get(y, x)[0] > 127) {
                    count++;
                }
            }
            maxCol = Math.max(maxCol, count);
            sumCol += count;
        }
        double avgCol = sumCol / Math.max(region.cols(), 1);
        if (maxCol <= avgCol * 2.0 || maxCol < h * 0.25) {
            return false;
        }
        Mat points = new Mat();
        Core.findNonZero(region, points);
        if (points.rows() < 8) {
            return false;
        }
        double minX = region.cols(), maxX = 0, minY = h, maxY = 0;
        for (int i = 0; i < points.rows(); i++) {
            double[] p = points.get(i, 0);
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
        }
        double ratio = (maxX - minX + 1) / Math.max(maxY - minY + 1, 1);
        return ratio < 0.72;
    }

    /** Trait vertical du « 1 » au centre du panneau (bande médiane). */
    private static boolean looksLikeOneInMiddle(Mat mask) {
        int w = mask.cols();
        int h = mask.rows();
        int x0 = (int) (w * 0.30);
        int x1 = (int) (w * 0.70);
        Mat middle = new Mat(mask, new Rect(x0, 0, x1 - x0, h));
        return hasVerticalStrokeInRegion(middle);
    }

    /** Détection agressive du « 9 » sur la moitié gauche du masque. */
    private static Character looksLikeNine(Mat mask) {
        int w = mask.cols();
        int h = mask.rows();
        Mat left = new Mat(mask, new Rect(0, 0, Math.max(w / 2, 1), h));
        Character c = analyzeRegion(left);
        if (c != null) {
            return c;
        }
        if (countHoles(left) >= 1) {
            Mat points = new Mat();
            Core.findNonZero(left, points);
            if (points.rows() >= 10) {
                double top = 0;
                for (int i = 0; i < points.rows(); i++) {
                    if (points.get(i, 0)[1] < h / 2.0) {
                        top++;
                    }
                }
                if (top / points.rows() > 0.42) {
                    return '9';
                }
            }
        }
        return null;
    }

    /**
     * Lit la vitesse en découpant le masque en bandes verticales (9 | 0 → 90).
     */
    private static String readSpeedFromMask(Mat mask, List<String> speedLabels) {
        String three = tryReadThreeDigitSpeed(mask, speedLabels);
        if (three != null) {
            return three;
        }

        List<int[]> bands = getDigitBands(mask);

        if (bands.size() >= 2) {
            if (bands.size() > 2) {
                bands = keepTwoWidestBands(bands);
            }
            String two = speedFromTwoDigits(
                    readDigitInBand(mask, bands.get(0)),
                    readDigitInBand(mask, bands.get(1)),
                    speedLabels
            );
            if (two != null) {
                if ("Speed_limit_40_km_h".equals(two) && looksLike110Layout(mask)
                        && speedLabels != null
                        && speedLabels.contains("Speed_limit_110_km_h")) {
                    return "Speed_limit_110_km_h";
                }
                return two;
            }
        }

        if (bands.size() == 1) {
            int w = mask.cols();
            int h = mask.rows();
            int mid = w / 2;
            Character d1 = analyzeRegion(new Mat(mask, new Rect(0, 0, mid, h)));
            Character d2 = analyzeRegion(new Mat(mask, new Rect(mid, 0, w - mid, h)));
            String two = speedFromTwoDigits(d1, d2, speedLabels);
            if (two != null) {
                return two;
            }
        }

        return null;
    }

    /** Lit 110 / 100 / 130 avant de réduire à deux bandes (évite 110 → 40). */
    private static String tryReadThreeDigitSpeed(Mat mask, List<String> speedLabels) {
        if (speedLabels != null && speedLabels.contains("Speed_limit_110_km_h")) {
            if (looksLike110Layout(mask)) {
                return "Speed_limit_110_km_h";
            }
        }

        List<int[]> bands = getDigitBands(mask);
        if (bands.size() >= 3) {
            if (bands.size() > 3) {
                bands = keepThreeLeftmostBands(bands);
            }
            Character d1 = readDigitInBand(mask, bands.get(0));
            Character d2 = readDigitInBand(mask, bands.get(1));
            Character d3 = readDigitInBand(mask, bands.get(2));
            if (d1 != null && d2 != null) {
                if (d3 == null) {
                    d3 = '0';
                }
                int speed = (d1 - '0') * 100 + (d2 - '0') * 10 + (d3 - '0');
                String label = toSpeedLabel(speed);
                if (label != null && speedLabels.contains(label)) {
                    return label;
                }
            }
        }

        int w = mask.cols();
        int h = mask.rows();
        if (w < 12) {
            return null;
        }
        int third = w / 3;
        Character t1 = analyzeRegion(new Mat(mask, new Rect(0, 0, third, h)));
        Character t2 = analyzeRegion(new Mat(mask, new Rect(third, 0, third, h)));
        Character t3 = analyzeRegion(new Mat(mask, new Rect(2 * third, 0, w - 2 * third, h)));
        if (t1 != null && t2 != null) {
            if (t3 == null) {
                t3 = '0';
            }
            int speed = (t1 - '0') * 100 + (t2 - '0') * 10 + (t3 - '0');
            String label = toSpeedLabel(speed);
            if (label != null && speedLabels.contains(label)) {
                return label;
            }
        }

        return null;
    }

    private static String speedFromTwoDigits(Character d1, Character d2, List<String> speedLabels) {
        if (d1 == null) {
            return null;
        }
        if (d2 == null) {
            d2 = '0';
        }
        int speed = (d1 - '0') * 10 + (d2 - '0');
        String label = toSpeedLabel(speed);
        if (label != null && speedLabels.contains(label)) {
            return label;
        }
        return null;
    }

    private static Character readFirstDigit(Mat mask, List<int[]> bands) {
        if (bands.size() >= 1) {
            return readDigitInBand(mask, bands.get(0));
        }
        int w = mask.cols();
        Mat left = new Mat(mask, new Rect(0, 0, (int) (w * 0.52), mask.rows()));
        return analyzeRegion(left);
    }

    private static Character readDigitInBand(Mat mask, int[] band) {
        int x0 = band[0];
        int x1 = band[1];
        if (x1 <= x0) {
            return null;
        }
        Mat region = new Mat(mask, new Rect(x0, 0, x1 - x0 + 1, mask.rows()));
        return analyzeRegion(region);
    }

    private static List<int[]> getDigitBands(Mat mask) {
        int cols = mask.cols();
        int rows = mask.rows();
        double threshold = rows * 0.05;

        List<int[]> segments = new ArrayList<int[]>();
        int start = -1;

        for (int x = 0; x < cols; x++) {
            int count = 0;
            for (int y = 0; y < rows; y++) {
                if (mask.get(y, x)[0] > 127) {
                    count++;
                }
            }

            if (count > threshold) {
                if (start < 0) {
                    start = x;
                }
            } else if (start >= 0) {
                if (x - start > 2) {
                    segments.add(new int[]{start, x - 1});
                }
                start = -1;
            }
        }

        if (start >= 0 && cols - start > 2) {
            segments.add(new int[]{start, cols - 1});
        }

        return segments;
    }

    private static List<int[]> keepTwoWidestBands(List<int[]> bands) {
        Collections.sort(bands, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return Integer.compare(b[1] - b[0], a[1] - a[0]);
            }
        });
        List<int[]> top = new ArrayList<int[]>();
        top.add(bands.get(0));
        top.add(bands.get(1));
        Collections.sort(top, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return Integer.compare(a[0], b[0]);
            }
        });
        return top;
    }

    private static List<int[]> keepThreeLeftmostBands(List<int[]> bands) {
        Collections.sort(bands, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return Integer.compare(a[0], b[0]);
            }
        });
        List<int[]> top = new ArrayList<int[]>();
        for (int i = 0; i < Math.min(3, bands.size()); i++) {
            top.add(bands.get(i));
        }
        return top;
    }

    private static Character analyzeRegion(Mat region) {
        Mat points = new Mat();
        Core.findNonZero(region, points);
        if (points.rows() < 10) {
            return null;
        }

        int holes = countHoles(region);

        double topInk = 0;
        double bottomInk = 0;
        double midY = region.rows() / 2.0;
        double minX = region.cols(), maxX = 0, minY = region.rows(), maxY = 0;

        for (int i = 0; i < points.rows(); i++) {
            double[] p = points.get(i, 0);
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
            if (p[1] < midY) {
                topInk++;
            } else {
                bottomInk++;
            }
        }

        double total = topInk + bottomInk;
        double topRatio = topInk / Math.max(total, 1);
        double ratio = (maxX - minX + 1) / Math.max(maxY - minY + 1, 1);

        // Trait vertical "1"
        if (ratio < 0.40) {
            return '1';
        }

        // "0" rond
        if (holes >= 1 && ratio > 0.55 && topRatio > 0.35 && topRatio < 0.58) {
            return '0';
        }

        // "9" : trou + plus d'encre en haut
        if (holes >= 1 && topRatio >= 0.44) {
            return '9';
        }

        // "8"
        if (holes >= 2) {
            return '8';
        }

        // "3" : pas de trou, moins d'encre en haut qu'un 5 (pas de barre horizontale haute)
        if (holes == 0 && topRatio < 0.46 && looksLikeThree(region, topRatio, ratio)) {
            return '3';
        }

        // "6" : trou + encre plutôt en bas
        if (holes >= 1 && topRatio < 0.42) {
            return '6';
        }

        // "5" : barre du haut → plus d'encre dans le tiers supérieur
        if (holes == 0 && ratio > 0.50 && topRatio >= 0.44 && topRatio < 0.54
                && looksLikeFive(region, topRatio)) {
            return '5';
        }

        // "3" repli (formes compactes sans barre haute)
        if (holes == 0 && topRatio < 0.50) {
            return '3';
        }

        // "2"
        if (holes == 0 && topRatio >= 0.48 && topRatio < 0.58) {
            return '2';
        }

        return null;
    }

    private static int countHoles(Mat region) {
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(
                region.clone(),
                contours,
                hierarchy,
                Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        if (hierarchy.empty()) {
            return 0;
        }

        int holes = 0;
        for (int i = 0; i < contours.size(); i++) {
            double[] h = hierarchy.get(0, i);
            if ((int) h[3] >= 0 && Imgproc.contourArea(contours.get(i)) > 8) {
                holes++;
            }
        }
        return holes;
    }

    private static String refineFromNeighbors(Mat signImage, String knnLabel,
                                              KNearest model, List<String> speedLabels, Mat mask) {
        Mat feature = SpeedLimitFeatures.toHogFromMat(signImage);
        if (feature.empty()) {
            return knnLabel;
        }

        Mat neighborResponses = new Mat();
        Mat dist = new Mat();
        model.findNearest(feature, 5, new Mat(), neighborResponses, dist);

        int knnId = speedLabels.indexOf(knnLabel);
        if (knnId < 0) {
            return knnLabel;
        }

        float dKnn = Float.MAX_VALUE;
        for (int i = 0; i < neighborResponses.rows(); i++) {
            if ((int) neighborResponses.get(i, 0)[0] == knnId) {
                dKnn = Math.min(dKnn, (float) dist.get(i, 0)[0]);
            }
        }

        for (int i = 0; i < neighborResponses.rows(); i++) {
            int nid = (int) neighborResponses.get(i, 0)[0];
            float d = (float) dist.get(i, 0)[0];
            if (nid == knnId || d > dKnn * 1.5f) {
                continue;
            }

            String alt = speedLabels.get(nid);
            String read = readSpeedFromMask(mask, speedLabels);
            if (read != null && read.equals(alt)) {
                return alt;
            }
        }

        return knnLabel;
    }

    private static String applyRulesThreeDigit(String label, Character middle) {
        if (middle == null) {
            return label;
        }
        char m = middle.charValue();
        if ("Speed_limit_100_km_h".equals(label) && m == '1') {
            return "Speed_limit_110_km_h";
        }
        if ("Speed_limit_110_km_h".equals(label) && m == '0') {
            return "Speed_limit_100_km_h";
        }
        if ("Speed_limit_40_km_h".equals(label) && m == '1') {
            return "Speed_limit_110_km_h";
        }
        return label;
    }

    private static String applyRulesTwoDigit(String label, Character first) {
        if (first == null) {
            return label;
        }
        int speed = parseSpeed(label);
        if (speed < 0 || speed >= 100) {
            return label;
        }
        int ones = speed % 10;
        int corrected = (first - '0') * 10 + ones;
        String alt = toSpeedLabel(corrected);
        return alt != null ? alt : label;
    }

    private static Character detectMiddleDigitByThirds(Mat mask) {
        int w = mask.cols();
        Mat mid = new Mat(mask, new Rect((int) (w * 0.30), 0, (int) (w * 0.40), mask.rows()));
        return analyzeRegion(mid);
    }

    private static String toSpeedLabel(int speed) {
        if (speed == 20 || speed == 30 || speed == 40 || speed == 50
                || speed == 60 || speed == 70 || speed == 80 || speed == 90
                || speed == 100 || speed == 110 || speed == 120 || speed == 130) {
            return "Speed_limit_" + speed + "_km_h";
        }
        return null;
    }

    private static int parseSpeed(String label) {
        try {
            return Integer.parseInt(
                    label.replace("Speed_limit_", "").replace("_km_h", ""));
        } catch (Exception e) {
            return -1;
        }
    }

    /**
     * Ne remplace le KNN par la lecture OCR que si la paire n'est pas ambiguë
     * (ex. 30↔50 où le « 3 » est souvent lu comme « 5 » sur photos réelles).
     */
    private static boolean shouldTrustDigitReadOverKnn(String knnLabel, String readLabel,
                                                       Mat mask, List<String> speedLabels) {
        int knnSpeed = parseSpeed(knnLabel);
        int readSpeed = parseSpeed(readLabel);
        if (readSpeed == 20 && knnSpeed >= 30 && knnSpeed <= 90) {
            return false;
        }
        if (!isAmbiguousSpeedPair(knnLabel, readLabel)) {
            return true;
        }
        if (speedLabels == null) {
            return false;
        }
        return isDigitReadConfident(mask, readLabel, speedLabels);
    }

    /** Même dizaine des unités, dizaines facilement confondues (3/5/8/9). */
    private static boolean isAmbiguousSpeedPair(String labelA, String labelB) {
        int a = parseSpeed(labelA);
        int b = parseSpeed(labelB);
        if (a < 0 || b < 0 || a >= 100 || b >= 100) {
            return false;
        }
        if (a % 10 != b % 10) {
            return false;
        }
        int tensA = a / 10;
        int tensB = b / 10;
        return (tensA == 3 && tensB == 5) || (tensA == 5 && tensB == 3)
                || (tensA == 3 && tensB == 9) || (tensA == 9 && tensB == 3)
                || (tensA == 7 && tensB == 3) || (tensA == 3 && tensB == 7)
                || (tensA == 8 && tensB == 9) || (tensA == 9 && tensB == 8);
    }

    /** Lecture à deux bandes distinctes + pas de contradiction 3/5 sur la 1re bande. */
    private static boolean isDigitReadConfident(Mat mask, String readLabel,
                                                List<String> speedLabels) {
        List<int[]> bands = getDigitBands(mask);
        if (bands.size() < 2) {
            return false;
        }
        if (bands.size() > 2) {
            bands = keepTwoWidestBands(bands);
        }

        Character d1 = readDigitInBand(mask, bands.get(0));
        Character d2 = readDigitInBand(mask, bands.get(1));
        if (d1 == null || d2 == null) {
            return false;
        }

        String check = speedFromTwoDigits(d1, d2, speedLabels);
        if (!readLabel.equals(check)) {
            return false;
        }

        int speed = parseSpeed(readLabel);
        if (speed < 20 || speed >= 100) {
            return true;
        }

        int tens = speed / 10;
        if (tens != 3 && tens != 5) {
            return true;
        }

        int x0 = bands.get(0)[0];
        int x1 = bands.get(0)[1];
        Mat firstBand = new Mat(mask, new Rect(x0, 0, x1 - x0 + 1, mask.rows()));
        if (tens == 5 && looksLikeThree(firstBand, -1, -1)) {
            return false;
        }
        if (tens == 3 && looksLikeFive(firstBand, -1)) {
            return false;
        }
        return true;
    }

    private static boolean looksLikeThree(Mat region, double topRatio, double ratio) {
        if (topRatio >= 0 && topRatio >= 0.46) {
            return false;
        }
        if (ratio >= 0 && ratio > 0.62) {
            return false;
        }
        return topThirdInkRatio(region) < 0.38;
    }

    private static boolean looksLikeFive(Mat region, double topRatio) {
        if (topRatio >= 0 && topRatio < 0.44) {
            return false;
        }
        return topThirdInkRatio(region) >= 0.36;
    }

    /** Part d'encre dans le tiers supérieur de la région (barre du 5). */
    private static double topThirdInkRatio(Mat region) {
        Mat points = new Mat();
        Core.findNonZero(region, points);
        if (points.rows() < 8) {
            return 0;
        }
        double top = 0;
        double limit = region.rows() / 3.0;
        for (int i = 0; i < points.rows(); i++) {
            if (points.get(i, 0)[1] < limit) {
                top++;
            }
        }
        return top / points.rows();
    }
}
