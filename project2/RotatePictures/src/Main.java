import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

@SuppressWarnings("WeakerAccess")
public class Main {
    //groundtruth
    private static final String folderIn =  "/Users/leobouraux/Downloads/groundtruth/";

    private static final String folderOut = folderIn + "DONE/";

    public static void main(String[] args) {
        try {
            File f = new File(folderIn);
            ArrayList<String> file_names = listFilenamesToPreprocess(f);
            dataCreator(file_names);
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }

    private static ArrayList<String> listFilenamesToPreprocess(final File folder) throws NullPointerException{
        ArrayList<String> files_names = new ArrayList<>();

        File[] files = folder.listFiles();
        if(files != null) {
            for (final File fileEntry : files) {
                if (fileEntry.isDirectory()) {
                    listFilenamesToPreprocess(fileEntry);
                } else {
                    files_names.add(fileEntry.getName());
                }
            }
        }
        return files_names;
    }

    public static void dataCreator(ArrayList<String> file_names) throws IOException {
        boolean isDONEfoldercreated = new File(folderOut).mkdir();
        if(!isDONEfoldercreated) System.out.println("DONE folder not created\n-----------------------\n");
        List<Integer> rotations =  Arrays.asList(0, 45, 90, 135, 180, 225, 270, 315);
        List<String> reverses = Arrays.asList("N", "H", "V");

        for (String photoName : file_names) {
            String imagePathFrom = folderIn + photoName;

            for (int rotation : rotations) {

                for (String reverse:reverses) {
                    String newPhotoName = insertString(photoName, "_"+rotation+"_"+reverse, photoName.length()-4);
                    String imagePathTo = folderOut + newPhotoName;
                    File f = new File(imagePathFrom);

                    System.out.println(imagePathFrom);
                    if (!f.canRead()) {
                        continue;
                    }
                    BufferedImage initImage = ImageIO.read(f);
                    if (initImage == null) {
                        continue;
                    }

                    BufferedImage rotatedImage = rotateImage(rotation, initImage);

                    if(rotation%10 != 0) {
                        BufferedImage croppedImage = rotatedImage.getSubimage(rotatedImage.getWidth() / 4, rotatedImage.getHeight() / 4, rotatedImage.getWidth() / 2, rotatedImage.getHeight() / 2);
                        Image newImage = croppedImage.getScaledInstance(initImage.getWidth(), initImage.getHeight(), Image.SCALE_DEFAULT);
                        rotatedImage = toBufferedImage(newImage);
                    }

                    switch (reverse) {
                        case "H":
                            // Flip the image horizontally
                            AffineTransform tx = AffineTransform.getScaleInstance(-1, 1);
                            tx.translate(-rotatedImage.getWidth(null), 0);
                            AffineTransformOp op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
                            rotatedImage = op.filter(rotatedImage, null);
                            ImageIO.write(rotatedImage, "png", new File(imagePathTo));
                            break;
                        case "V":
                            // Flip the image vertically
                            AffineTransform tx2 = AffineTransform.getScaleInstance(1, -1);
                            tx2.translate(0, -rotatedImage.getHeight(null));
                            AffineTransformOp op2 = new AffineTransformOp(tx2, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
                            rotatedImage = op2.filter(rotatedImage, null);
                            ImageIO.write(rotatedImage, "png", new File(imagePathTo));
                            break;
                        default:
                            ImageIO.write(rotatedImage, "png", new File(imagePathTo));
                    }

                }


            }
        }
    }

    private static BufferedImage rotateImage(int rotation, BufferedImage initImage) {
        final double rads = Math.toRadians(rotation);
        final double sin = Math.abs(Math.sin(rads));
        final double cos = Math.abs(Math.cos(rads));
        final int w = (int) Math.floor(initImage.getWidth() * cos + initImage.getHeight() * sin);
        final int h = (int) Math.floor(initImage.getHeight() * cos + initImage.getWidth() * sin);
        final BufferedImage rotatedImage = new BufferedImage(w, h, initImage.getType());
        final AffineTransform at = new AffineTransform();
        at.translate(w / 2, h / 2);
        at.rotate(rads,0, 0);
        at.translate(-initImage.getWidth() / 2, -initImage.getHeight() / 2);
        final AffineTransformOp rotateOp = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
        rotateOp.filter(initImage,rotatedImage);
        return rotatedImage;
    }

    public static String insertString(String init, String toInsert, int index) {
        String a = init.substring(0,index);
        String b = init.substring(index);
        return a + toInsert + b;
    }

    public static BufferedImage toBufferedImage(Image img)
    {
        if (img instanceof BufferedImage)
        {
            return (BufferedImage) img;
        }

        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }

}


