import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Tokenizer {
    private Pattern pattern;
    private opennlp.tools.tokenize.Tokenizer tokenizer;

    public Tokenizer() {
        pattern = Pattern.compile("[\\p{L}\\p{M}]([/" +
                            ","+
                            "\u2010" +
                            "\u2011" +
                            "\u2012" +
                            "\u2013" +
                            "\u2014" +
                            "\u2015" +
                            "\uFF0D" +
                            "\u002D])[\\p{L}\\p{M}]");
        InputStream modelIn = null;
        try {
//            modelIn = new FileInputStream("lib/en-token.bin");
            modelIn = Tokenizer.class.getResourceAsStream("/en-token.bin");
            tokenizer = new TokenizerME(new TokenizerModel(modelIn));
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        finally {
            if (modelIn != null) {
                try {
                    modelIn.close();
                }
                catch (IOException e) {
                }
            }
        }
    }

    public List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<String>();

        for (String tokenText : tokenizer.tokenize(text)) {
            boolean progress;
            do {
                progress = false;
                Matcher matcher = pattern.matcher(tokenText);
                if (matcher.find()) {
                    progress = true;
                    int start = matcher.start(1);
                    int len = matcher.end(1) - start;
                    if (start > 0) {
                        tokens.add(tokenText.substring(0, start));
                        tokenText = tokenText.substring(start);
                    }
                    tokens.add(tokenText.substring(0, len));
                    tokenText = tokenText.substring(len);
                }
            } while (progress);
            if (!tokenText.isEmpty()) {
                tokens.add(tokenText);
            }
        }
        return tokens;
    }

    public static void main(String[] args) throws IOException {
        String text = "Extra Duty Pay/Overtime/Pay/sd For Support Personnel";
        Tokenizer t = new Tokenizer();

        for (String tokenText : t.tokenize(text)) {
            System.out.println(tokenText);
        }
    }
}
