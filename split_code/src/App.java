import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Stack;

public class App {
    public static String sourceDirectory;
    public static String targetDirectory;
    public static PrintWriter indexer;

    private double split;
    private String[] uploaders;

    public App() {
        App.sourceDirectory = "D:/GitHub/meng-ucalgary/enel-645-project/dataset";
        App.targetDirectory = "D:/GitHub/meng-ucalgary/enel-645-project/dataset_test";

        this.split = 0.2;
        this.uploaders = new String[] { "bg", "db", "jf", "kg", "ml", "ts" };

        try {
            App.indexer = new PrintWriter(new BufferedWriter(new FileWriter("indexer.log", true)), true);
        }

        catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Generates random string to length 10
     */
    @SuppressWarnings("unused")
    private String generateRandomString() {
        int leftLimit = 97;
        int rightLimit = 122;
        int targetStringLength = 10;
        Random random = new Random();

        String generatedString = random.ints(leftLimit, rightLimit + 1)
                .limit(targetStringLength)
                .collect(StringBuilder::new, StringBuilder::appendCodePoint,
                        StringBuilder::append)
                .toString();

        return generatedString;
    }

    /**
     * splits the files by uploader into List of Stack
     */
    public List<Stack<File>> uploaderUploadCount(File[] list) {
        List<Stack<File>> uploaderSplit = new ArrayList<>();

        for (int i = 0; i < this.uploaders.length; i++) {
            uploaderSplit.add(new Stack<>());
        }

        for (File f : list) {
            if (f.getName().startsWith(this.uploaders[0])) {
                uploaderSplit.get(0).push(f);
            }

            else if (f.getName().startsWith(this.uploaders[1])) {
                uploaderSplit.get(1).push(f);
            }

            else if (f.getName().startsWith(this.uploaders[2])) {
                uploaderSplit.get(2).push(f);
            }

            else if (f.getName().startsWith(this.uploaders[3])) {
                uploaderSplit.get(3).push(f);
            }

            else if (f.getName().startsWith(this.uploaders[4])) {
                uploaderSplit.get(4).push(f);
            }

            else {
                uploaderSplit.get(5).push(f);
            }
        }

        return uploaderSplit;
    }

    /**
     * Randomly picks a uploader index to move next file from
     */
    public int randomUploader(List<Stack<File>> uploaderSplit) {
        Random r = new Random();
        int x = r.nextInt(this.uploaders.length);

        while (uploaderSplit.get(x).size() == 0) {
            x = r.nextInt(6);
        }

        return x;
    }

    /**
     * Moves file from dataset to dataset_test
     */
    private void mover(File f) {
        String picName = f.getName();
        String cardClass = f.getParentFile().getName();

        File renamedFile = new File(App.targetDirectory + "/" + cardClass + "/" + picName);

        try {
            f.renameTo(renamedFile);
            indexer.printf("\"%s\", \"%s\"%n", f.getAbsolutePath(), renamedFile.getAbsolutePath());
        }

        catch (Exception e) {
            indexer.printf("\"%s\" could not be renamed to \"%s\"%n", f.getAbsolutePath(),
                    renamedFile.getAbsolutePath());
        }
    }

    /**
     * Loops over all files and folders
     */
    private void looper(String filePath) {
        File currentFile = new File(filePath);
        File[] cardTypeList = currentFile.listFiles();

        // this might help in case of permission denials
        if (cardTypeList == null) {
            System.out.printf("%n%nCould not find or enlist directory %s", filePath);
        }

        for (File cardType : cardTypeList) {
            // windows default mandatory skip list
            if ((cardType.getName().contains("$RECYCLE.BIN"))
                    || (cardType.getName().equals("System Volume Information"))) {
                continue;
            }

            // listing all cards
            File[] cardPics = cardType.listFiles();

            // total cards per type
            int totalCards = cardPics.length;

            // splitting test cards
            int testCards = (int) Math.round(((double) totalCards * this.split));

            // System.out.printf("%n %15s, %d, %d", cardType.getName(), totalCards,
            // testCards);

            // split files by uploader
            List<Stack<File>> uploaderSplit = uploaderUploadCount(cardPics);

            // first move one card each from 6 uploader
            for (int i = 0; i < uploaderSplit.size(); i++) {
                Stack<File> uploaderFiles = uploaderSplit.get(i);

                if (uploaderFiles.size() != 0) {
                    this.mover(uploaderFiles.pop());
                    testCards--;
                }
            }

            // now move cards randomly
            while (testCards != 0) {
                int nextTurn = this.randomUploader(uploaderSplit);

                if (uploaderSplit.get(nextTurn).size() != 0) {
                    this.mover(uploaderSplit.get(nextTurn).pop());
                    testCards--;
                }
            }

        }
    }

    public static void main(String[] args) throws Exception {
        App app = new App();
        app.looper(App.sourceDirectory);
    }
}
