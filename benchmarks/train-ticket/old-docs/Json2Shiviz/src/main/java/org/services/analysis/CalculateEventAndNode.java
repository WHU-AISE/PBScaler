package org.services.analysis;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CalculateEventAndNode {

    private static String ROOT_DIR = "C:\\Users\\dingding\\Desktop\\tse debug exp\\F20\\shiviz\\4 scope failure elements\\shiviz_trace";
    private static String caseDirectory = "case1";
    private static String fileName = "all.txt";
    private static Pattern patternEvent = Pattern.compile("\\{traceId=.*spanId=.*event=.*\\}");
    private static Pattern patternNode = Pattern.compile("host=(\\S*),");
    private static HashSet<String> nodes = new HashSet<String>();

    public static void main(String[] args){
        try {
            int count = 0;
            File file = new File(String.format("%s\\%s\\%s",ROOT_DIR,caseDirectory,fileName));
            InputStreamReader reader = new InputStreamReader(new FileInputStream(file));
            BufferedReader br = new BufferedReader(reader);
            String line = null;
            while((line = br.readLine())!=null){
                Matcher matcherEvent = patternEvent.matcher(line);
                if(matcherEvent.find()) {
                    System.out.println(line);
                    count ++;
                    Matcher matcherNode = patternNode.matcher(line);
                    if(matcherNode.find()){
                        nodes.add(matcherNode.group(1));
                    }
                }
            }
            System.out.println(nodes);
            System.out.println(String.format("The number of event is [%d]", count));
            System.out.println(String.format("The number of node is [%d]", nodes.size()));
            reader.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }

    }

}
