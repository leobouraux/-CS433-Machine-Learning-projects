public class TEST {
    public static void main(String[] args) {
        String s = "java.png";

        System.out.println(insert(s, "LOL", s.length()-4));
    }

    public static String insert(String init, String toInsert, int index) {
        String a = init.substring(0,index);
        String b = init.substring(index);
        return a + toInsert + b;
    }
}
