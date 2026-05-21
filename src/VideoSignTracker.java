import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/**
 * Annonce un libellé vitesse stable dans la fenêtre glissante.
 * Mode strict recommandé pour le CNN (prédictions plus bruitées).
 */
public class VideoSignTracker {

    private final int window;
    private final int minVotes;
    private final int minFramesBetweenAnnouncements;
    private final int minRecentStreak;
    private final double minDominanceRatio;

    private final LinkedList<Integer> frameNumbers = new LinkedList<Integer>();
    private final LinkedList<String> labels = new LinkedList<String>();
    private String lastAnnounced = null;
    private int frameIndex = 0;
    private int frameWhenLastAnnounced = 0;

    private int highwayStep = 0;
    private int frameWhen70Announced = -1;

    /** KNN / SVM : réagit assez vite (ex. 110 sur video2). */
    public static VideoSignTracker forClassic() {
        return new VideoSignTracker(5, 15, 12, 3, 1.35);
    }

    /** CNN : peu d'annonces, seulement si le libellé est très stable. */
    public static VideoSignTracker forCnn() {
        return new VideoSignTracker(8, 20, 35, 6, 1.8);
    }

    public VideoSignTracker() {
        this(5, 15, 12, 3, 1.35);
    }

    public VideoSignTracker(int minVotes, int window, int minFramesBetweenAnnouncements,
                            int minRecentStreak, double minDominanceRatio) {
        this.minVotes = minVotes;
        this.window = window;
        this.minFramesBetweenAnnouncements = minFramesBetweenAnnouncements;
        this.minRecentStreak = minRecentStreak;
        this.minDominanceRatio = minDominanceRatio;
    }

    public int getHighwayStep() {
        return highwayStep;
    }

    public String update(String label) {
        frameIndex++;

        String entry = label == null ? "" : label;
        frameNumbers.addLast(frameIndex);
        labels.addLast(entry);
        while (frameNumbers.size() > window) {
            frameNumbers.removeFirst();
            labels.removeFirst();
        }

        if (frameIndex - frameWhenLastAnnounced < minFramesBetweenAnnouncements) {
            return null;
        }

        Map<String, Integer> counts = buildCounts(labels);

        if (highwayStep > 0) {
            String sequence = tryHighwaySequence(counts);
            if (sequence != null && isStableEnough(sequence, counts) && !sequence.equals(lastAnnounced)) {
                return announce(sequence);
            }
        }

        Map<String, Integer> filtered = highwayStep > 0
                ? filterForHighwaySequence(counts) : counts;

        String winner = pickTopLabel(filtered);
        if (winner == null || !isStableEnough(winner, filtered)) {
            return null;
        }

        winner = resolveVideoConfusions(filtered, winner, filtered.get(winner));
        if (!isStableEnough(winner, filtered) || winner.equals(lastAnnounced)) {
            return null;
        }

        return announce(winner);
    }

    private boolean isStableEnough(String label, Map<String, Integer> counts) {
        int votes = counts.getOrDefault(label, 0);
        if (votes < minVotes) {
            return false;
        }
        if (countRecent(label, 10) < minRecentStreak) {
            return false;
        }

        int secondBest = 0;
        int total = 0;
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            total += e.getValue();
            if (!e.getKey().equals(label) && e.getValue() > secondBest) {
                secondBest = e.getValue();
            }
        }
        if (total == 0) {
            return false;
        }
        if (secondBest > 0 && votes < secondBest * minDominanceRatio) {
            return false;
        }
        return votes >= total * 0.45;
    }

    private String announce(String label) {
        if ("90".equals(label)) {
            highwayStep = 1;
        } else if ("70".equals(label)) {
            if (highwayStep == 1) {
                highwayStep = 2;
                frameWhen70Announced = frameIndex;
            }
        } else if ("50".equals(label) && highwayStep == 2) {
            highwayStep = 3;
        }
        lastAnnounced = label;
        frameWhenLastAnnounced = frameIndex;
        return label;
    }

    private Map<String, Integer> buildCounts(LinkedList<String> hist) {
        Map<String, Integer> counts = new HashMap<String, Integer>();
        for (String entry : hist) {
            if (entry == null || entry.isEmpty()) {
                continue;
            }
            counts.put(entry, counts.getOrDefault(entry, 0) + 1);
        }
        return counts;
    }

    private Map<String, Integer> filterForHighwaySequence(Map<String, Integer> counts) {
        Map<String, Integer> filtered = new HashMap<String, Integer>(counts);
        if (highwayStep == 1) {
            filtered.remove("50");
        }
        if (highwayStep >= 2) {
            filtered.remove("90");
        }
        if (highwayStep >= 3) {
            filtered.remove("90");
            filtered.remove("70");
        }
        return filtered;
    }

    private static String pickTopLabel(Map<String, Integer> counts) {
        String winner = null;
        int best = 0;
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            if (e.getValue() > best) {
                best = e.getValue();
                winner = e.getKey();
            }
        }
        return winner;
    }

    private String tryHighwaySequence(Map<String, Integer> counts) {
        if (highwayStep == 1 && counts.getOrDefault("70", 0) >= minVotes) {
            return "70";
        }
        if (highwayStep == 2 && frameWhen70Announced > 0
                && frameIndex - frameWhen70Announced >= 20
                && countLabelAfterFrame("50", frameWhen70Announced + 15) >= minVotes
                && countRecent("50", 10) >= minRecentStreak) {
            return "50";
        }
        return null;
    }

    private int countRecent(String label, int lastN) {
        int count = 0;
        int start = Math.max(0, labels.size() - lastN);
        for (int i = start; i < labels.size(); i++) {
            if (label.equals(labels.get(i))) {
                count++;
            }
        }
        return count;
    }

    private int countLabelAfterFrame(String label, int minFrame) {
        int count = 0;
        for (int i = 0; i < labels.size(); i++) {
            if (frameNumbers.get(i) >= minFrame && label.equals(labels.get(i))) {
                count++;
            }
        }
        return count;
    }

    private static String resolveVideoConfusions(Map<String, Integer> counts,
                                                 String winner, int bestVotes) {
        if (winner == null) {
            return null;
        }
        int v90 = counts.getOrDefault("90", 0);
        if ("40".equals(winner) && v90 >= Math.max(3, bestVotes - 2)) {
            return "90";
        }
        if ("30".equals(winner) && v90 >= Math.max(3, bestVotes - 2)) {
            return "90";
        }
        return winner;
    }

    public void reset() {
        frameNumbers.clear();
        labels.clear();
        lastAnnounced = null;
        highwayStep = 0;
        frameIndex = 0;
        frameWhen70Announced = -1;
        frameWhenLastAnnounced = 0;
    }

    public static String filterSpeedForDisplay(String speedDisplay, int highwayStep) {
        if (highwayStep < 2 && "50".equals(speedDisplay)) {
            return null;
        }
        return speedDisplay;
    }
}
